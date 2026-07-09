import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Depthwise conv2d optimized CUDA kernel - Iter 6 (FINAL)
# Based on iter-5 (best at 2.62ms). Key insight:
# - NVEC=4 in X + NVEC=2 in Y = 8 outputs per thread, 32x8=256 threads
# - Float4 vectorized loads for smem filling (as in iter-5)
# - Float4 aligned stores where possible (output ow aligned by 128)
# - Goal: push below 2.62ms towards the ~2.48ms floor
#
# New idea: 2-plane parallel processing
# - Each block handles 2 consecutive NC planes (n, c) and (n, c+1)
# - Doubles the arithmetic per block, better amortizes overhead
# - However requires 2x smem -> might hurt occupancy
# - Try instead: keep 1 plane per block but use 2x unrolled loading

_depthwise_conv_src = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// ==========================================================================
// Best variant: NVEC=4 X + NVEC=2 Y, 128x16 tile, float4 smem loads
// Block: 32x8=256 threads; Output: 128x16; Smem: 130x18 = 2340 floats = 9.4KB
// ==========================================================================
#define THX  32
#define THY   8
#define NVX   4   // outputs per thread in X
#define NVY   2   // outputs per thread in Y
#define OUT_W (THX * NVX)     // 128
#define OUT_H (THY * NVY)     // 16
#define IN_W  (OUT_W + 2)     // 130
#define IN_H  (OUT_H + 2)     // 18

__global__ void depthwise_conv2d_3x3_nvxy(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int OH, int OW
) {
    // smem: IN_H x (IN_W + 2) = 18 x 132 = 2376 floats = 9.5KB
    __shared__ float sdata[IN_H][IN_W + 2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int ow_base = blockIdx.x * OUT_W;   // multiples of 128
    int oh_base = blockIdx.y * OUT_H;   // multiples of 16
    int nc = blockIdx.z, n = nc / C, c = nc % C;

    // Load weights into registers
    const float* wp = weight + c * 9;
    float w00=wp[0],w01=wp[1],w02=wp[2];
    float w10=wp[3],w11=wp[4],w12=wp[5];
    float w20=wp[6],w21=wp[7],w22=wp[8];
    const float* inp = input + (n * C + c) * (H * W);

    // Load smem with float4 vectorized loads
    // First 128 cols of each of 18 rows: 18 * 32 = 576 float4 loads
    // Last 2 cols (128, 129) of each of 18 rows: 36 scalar loads
    int flat_tid = ty * THX + tx;  // 0..255

    // Float4 loads: total 576, 256 threads -> ~2.25 each
    int total_f4 = IN_H * 32;  // 576
    #pragma unroll 3
    for (int i = flat_tid; i < total_f4; i += THX * THY) {
        int row = i / 32, col4 = i % 32;
        int ih = oh_base + row, iw = ow_base + col4 * 4;
        float4 val = {0,0,0,0};
        if ((unsigned)ih < (unsigned)H) {
            if ((unsigned)(iw + 3) < (unsigned)W) {
                val = *((const float4*)(inp + ih * W + iw));
            } else {
                for (int k = 0; k < 4 && (unsigned)(iw+k) < (unsigned)W; k++)
                    ((float*)&val)[k] = inp[ih*W+iw+k];
            }
        }
        sdata[row][col4*4    ] = val.x;
        sdata[row][col4*4 + 1] = val.y;
        sdata[row][col4*4 + 2] = val.z;
        sdata[row][col4*4 + 3] = val.w;
    }
    // Scalar loads for trailing cols (128, 129)
    if (flat_tid < IN_H * 2) {
        int row = flat_tid / 2, col = 128 + (flat_tid % 2);
        int ih = oh_base + row, iw = ow_base + col;
        sdata[row][col] = ((unsigned)ih<(unsigned)H && (unsigned)iw<(unsigned)W)
                          ? inp[ih*W+iw] : 0.f;
    }

    __syncthreads();

    // Each thread computes NVY=2 rows x NVX=4 cols = 8 outputs
    int sx = tx * NVX;  // base smem col

    #pragma unroll
    for (int vy = 0; vy < NVY; vy++) {
        int oh = oh_base + ty * NVY + vy;
        int ow0 = ow_base + tx * NVX;
        if (oh >= OH) break;

        int sy = ty * NVY + vy;  // smem row for kernel row 0

        float s0=0,s1=0,s2=0,s3=0;

        // Row sy
        float r0_0=sdata[sy  ][sx  ],r0_1=sdata[sy  ][sx+1],r0_2=sdata[sy  ][sx+2];
        float r0_3=sdata[sy  ][sx+3],r0_4=sdata[sy  ][sx+4],r0_5=sdata[sy  ][sx+5];
        s0+=w00*r0_0+w01*r0_1+w02*r0_2; s1+=w00*r0_1+w01*r0_2+w02*r0_3;
        s2+=w00*r0_2+w01*r0_3+w02*r0_4; s3+=w00*r0_3+w01*r0_4+w02*r0_5;

        // Row sy+1
        float r1_0=sdata[sy+1][sx  ],r1_1=sdata[sy+1][sx+1],r1_2=sdata[sy+1][sx+2];
        float r1_3=sdata[sy+1][sx+3],r1_4=sdata[sy+1][sx+4],r1_5=sdata[sy+1][sx+5];
        s0+=w10*r1_0+w11*r1_1+w12*r1_2; s1+=w10*r1_1+w11*r1_2+w12*r1_3;
        s2+=w10*r1_2+w11*r1_3+w12*r1_4; s3+=w10*r1_3+w11*r1_4+w12*r1_5;

        // Row sy+2
        float r2_0=sdata[sy+2][sx  ],r2_1=sdata[sy+2][sx+1],r2_2=sdata[sy+2][sx+2];
        float r2_3=sdata[sy+2][sx+3],r2_4=sdata[sy+2][sx+4],r2_5=sdata[sy+2][sx+5];
        s0+=w20*r2_0+w21*r2_1+w22*r2_2; s1+=w20*r2_1+w21*r2_2+w22*r2_3;
        s2+=w20*r2_2+w21*r2_3+w22*r2_4; s3+=w20*r2_3+w21*r2_4+w22*r2_5;

        if (bias) { float b=bias[c]; s0+=b;s1+=b;s2+=b;s3+=b; }

        float* outp = output + ((n*C+c)*OH+oh)*OW + ow0;
        int rem = OW - ow0;
        if(rem>=4){outp[0]=s0;outp[1]=s1;outp[2]=s2;outp[3]=s3;}
        else{if(rem>0)outp[0]=s0;if(rem>1)outp[1]=s1;if(rem>2)outp[2]=s2;}
    }
}

// General stride/pad 3x3 (32x8 tile, scalar)
__global__ void depthwise_conv2d_3x3_general(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int N,int C,int H,int W,int OH,int OW,int sh,int sw,int ph,int pw
) {
    __shared__ float sg[10][36];
    int tx=threadIdx.x,ty=threadIdx.y;
    int owb=blockIdx.x*32,ohb=blockIdx.y*8,nc=blockIdx.z,n=nc/C,c=nc%C;
    const float* wp=weight+c*9;
    float w00=wp[0],w01=wp[1],w02=wp[2],w10=wp[3],w11=wp[4],w12=wp[5],w20=wp[6],w21=wp[7],w22=wp[8];
    const float* inp=input+(n*C+c)*(H*W);
    int flat=ty*32+tx;
    for(int i=flat;i<10*34;i+=256){
        int si=i/34,sj=i%34;int ih=ohb*sh-ph+si,iw=owb*sw-pw+sj;
        sg[si][sj]=((unsigned)ih<(unsigned)H&&(unsigned)iw<(unsigned)W)?inp[ih*W+iw]:0.f;
    }
    __syncthreads();
    int oh=ohb+ty,ow=owb+tx;
    if(oh<OH&&ow<OW){
        int sy=ty*sh,sx=tx*sw;
        float sum=w00*sg[sy][sx]+w01*sg[sy][sx+1]+w02*sg[sy][sx+2]
                 +w10*sg[sy+1][sx]+w11*sg[sy+1][sx+1]+w12*sg[sy+1][sx+2]
                 +w20*sg[sy+2][sx]+w21*sg[sy+2][sx+1]+w22*sg[sy+2][sx+2];
        if(bias)sum+=bias[c];
        output[((n*C+c)*OH+oh)*OW+ow]=sum;
    }
}

__global__ void depthwise_conv2d_general(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int N,int C,int H,int W,int OH,int OW,int KH,int KW,int sh,int sw,int ph,int pw
) {
    int ow=blockIdx.x*blockDim.x+threadIdx.x,oh=blockIdx.y*blockDim.y+threadIdx.y;
    int nc=blockIdx.z,n=nc/C,c=nc%C;
    if(ow>=OW||oh>=OH)return;
    const float* wp=weight+c*KH*KW;const float* inp=input+(n*C+c)*H*W;
    float sum=0.f;
    for(int kh=0;kh<KH;kh++){int ih=oh*sh-ph+kh;if((unsigned)ih>=(unsigned)H)continue;
        for(int kw=0;kw<KW;kw++){int iw=ow*sw-pw+kw;if((unsigned)iw>=(unsigned)W)continue;
            sum+=wp[kh*KW+kw]*__ldg(&inp[ih*W+iw]);}}
    if(bias)sum+=bias[c];
    output[((n*C+c)*OH+oh)*OW+ow]=sum;
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input, torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding
) {
    TORCH_CHECK(input.is_cuda()&&input.dtype()==torch::kFloat32&&input.is_contiguous());
    int N=input.size(0),C=input.size(1),H=input.size(2),W=input.size(3);
    int KH=weight.size(2),KW=weight.size(3);
    int sh=stride[0],sw=stride[1],ph=padding[0],pw=padding[1];
    int OH=(H+2*ph-KH)/sh+1, OW=(W+2*pw-KW)/sw+1;
    auto out=torch::empty({N,C,OH,OW},input.options());
    auto w=weight.contiguous().view({C,KH*KW});
    const float* bp=nullptr; torch::Tensor bt;
    if(bias.has_value()&&bias.value().defined()){bt=bias.value().contiguous();bp=bt.data_ptr<float>();}

    if(KH==3&&KW==3&&sh==1&&sw==1&&ph==0&&pw==0){
        dim3 block(THX,THY);
        dim3 grid((OW+OUT_W-1)/OUT_W,(OH+OUT_H-1)/OUT_H,N*C);
        depthwise_conv2d_3x3_nvxy<<<grid,block>>>(
            input.data_ptr<float>(),w.data_ptr<float>(),bp,out.data_ptr<float>(),N,C,H,W,OH,OW);
    } else if(KH==3&&KW==3){
        dim3 block(32,8);dim3 grid((OW+31)/32,(OH+7)/8,N*C);
        depthwise_conv2d_3x3_general<<<grid,block>>>(
            input.data_ptr<float>(),w.data_ptr<float>(),bp,out.data_ptr<float>(),N,C,H,W,OH,OW,sh,sw,ph,pw);
    } else {
        dim3 block(32,8);dim3 grid((OW+31)/32,(OH+7)/8,N*C);
        depthwise_conv2d_general<<<grid,block>>>(
            input.data_ptr<float>(),w.data_ptr<float>(),bp,out.data_ptr<float>(),N,C,H,W,OH,OW,KH,KW,sh,sw,ph,pw);
    }
    cudaError_t err=cudaGetLastError();
    if(err!=cudaSuccess)throw std::runtime_error(std::string("CUDA error: ")+cudaGetErrorString(err));
    return out;
}
"""

_depthwise_conv_decl = r"""
#include <torch/extension.h>
#include <vector>

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input, torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    std::vector<int64_t> stride, std::vector<int64_t> padding
);
"""

_depthwise_ext = load_inline(
    name="depthwise_conv2d_ext_v10",
    cpp_sources=_depthwise_conv_decl,
    cuda_sources=_depthwise_conv_src,
    functions=["depthwise_conv2d_forward"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],
)


class Model(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stride = [self.conv2d.stride[0], self.conv2d.stride[1]]
        padding = [self.conv2d.padding[0], self.conv2d.padding[1]]
        bias = self.conv2d.bias
        return _depthwise_ext.depthwise_conv2d_forward(
            x, self.conv2d.weight, bias, stride, padding
        )
