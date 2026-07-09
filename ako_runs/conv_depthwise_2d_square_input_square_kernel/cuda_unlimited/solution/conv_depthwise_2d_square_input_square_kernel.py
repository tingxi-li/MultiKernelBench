import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Depthwise conv2d optimized CUDA kernel - Iter 4
# Strategy: Try two configurations and pick best:
# A) 32x8 NVEC=4 (iter-2 baseline) but with cp.async pipeline for input loading
# B) 32x4 NVEC=4 with 4 output rows per thread (thread coarsening in Y)
#    - Each thread handles 4 output rows x 4 output cols = 16 outputs
#    - Block: 32x4 = 128 threads; Tile: 128x16 output; Smem: 18x130
# Hypothesis: Y-direction coarsening reduces smem loads (3x more data reuse per row read)

_depthwise_conv_src = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Config A: NVEC_X=4, NVEC_Y=2 (each thread: 4 cols x 2 rows = 8 outputs)
// Block: 32x8=256 threads, Tile: 128x16 output, Smem: 130x18 = 2340 floats = 9.4KB
// Benefit: 2x data reuse for row-shared weights, fewer synchronizations
#define THX_A  32
#define THY_A   8
#define NVEC_X  4
#define NVEC_Y  2   // Y-direction unrolling per thread
#define OUT_W_A (THX_A * NVEC_X)       // 128
#define OUT_H_A (THY_A * NVEC_Y)       // 16
#define IN_W_A  (OUT_W_A + 2)           // 130
#define IN_H_A  (OUT_H_A + 2)           // 18

__global__ void depthwise_conv2d_3x3_coarse_y(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int OH, int OW
) {
    // smem: 18 x (130 + 2) = 18 x 132 = 2376 floats = 9.5KB
    __shared__ float sdata[IN_H_A][IN_W_A + 2];

    int tx = threadIdx.x;  // 0..31
    int ty = threadIdx.y;  // 0..7
    int ow_base = blockIdx.x * OUT_W_A;   // multiples of 128
    int oh_base = blockIdx.y * OUT_H_A;   // multiples of 16
    int nc = blockIdx.z;
    int n = nc / C, c = nc % C;

    // Load weights
    const float* wp = weight + c * 9;
    float w00=wp[0],w01=wp[1],w02=wp[2],w10=wp[3],w11=wp[4],w12=wp[5],w20=wp[6],w21=wp[7],w22=wp[8];

    const float* inp = input + (n * C + c) * (H * W);

    // Load smem: IN_H_A * IN_W_A = 18 * 130 = 2340 cells
    // 256 threads each load ~9.2 cells
    int flat_tid = ty * THX_A + tx;
    int total = IN_H_A * IN_W_A;

    #pragma unroll 10
    for (int i = flat_tid; i < total; i += THX_A * THY_A) {
        int si = i / IN_W_A, sj = i % IN_W_A;
        int ih = oh_base + si, iw = ow_base + sj;
        float val = ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                    ? __ldg(&inp[ih * W + iw]) : 0.f;
        sdata[si][sj] = val;
    }

    __syncthreads();

    int sx = tx * NVEC_X;  // base col in smem

    // Each thread handles 2 output rows (ty*2 and ty*2+1) x 4 output cols
    #pragma unroll
    for (int vy = 0; vy < NVEC_Y; vy++) {
        int oh = oh_base + ty * NVEC_Y + vy;
        int sy = ty * NVEC_Y + vy;  // smem row for kernel row 0

        if (oh >= OH) break;

        int ow0 = ow_base + tx * NVEC_X;

        float s0=0,s1=0,s2=0,s3=0;

        // Row sy (kernel row 0)
        float r0_0=sdata[sy    ][sx],r0_1=sdata[sy    ][sx+1],r0_2=sdata[sy    ][sx+2];
        float r0_3=sdata[sy    ][sx+3],r0_4=sdata[sy    ][sx+4],r0_5=sdata[sy    ][sx+5];
        s0 += w00*r0_0+w01*r0_1+w02*r0_2;
        s1 += w00*r0_1+w01*r0_2+w02*r0_3;
        s2 += w00*r0_2+w01*r0_3+w02*r0_4;
        s3 += w00*r0_3+w01*r0_4+w02*r0_5;

        // Row sy+1 (kernel row 1)
        float r1_0=sdata[sy+1][sx],r1_1=sdata[sy+1][sx+1],r1_2=sdata[sy+1][sx+2];
        float r1_3=sdata[sy+1][sx+3],r1_4=sdata[sy+1][sx+4],r1_5=sdata[sy+1][sx+5];
        s0 += w10*r1_0+w11*r1_1+w12*r1_2;
        s1 += w10*r1_1+w11*r1_2+w12*r1_3;
        s2 += w10*r1_2+w11*r1_3+w12*r1_4;
        s3 += w10*r1_3+w11*r1_4+w12*r1_5;

        // Row sy+2 (kernel row 2)
        float r2_0=sdata[sy+2][sx],r2_1=sdata[sy+2][sx+1],r2_2=sdata[sy+2][sx+2];
        float r2_3=sdata[sy+2][sx+3],r2_4=sdata[sy+2][sx+4],r2_5=sdata[sy+2][sx+5];
        s0 += w20*r2_0+w21*r2_1+w22*r2_2;
        s1 += w20*r2_1+w21*r2_2+w22*r2_3;
        s2 += w20*r2_2+w21*r2_3+w22*r2_4;
        s3 += w20*r2_3+w21*r2_4+w22*r2_5;

        if (bias != nullptr) {
            float b = bias[c];
            s0+=b; s1+=b; s2+=b; s3+=b;
        }

        float* outp = output + ((n*C+c)*OH+oh)*OW + ow0;
        int rem = OW - ow0;
        if (rem >= 4) { outp[0]=s0; outp[1]=s1; outp[2]=s2; outp[3]=s3; }
        else {
            if (rem > 0) outp[0]=s0;
            if (rem > 1) outp[1]=s1;
            if (rem > 2) outp[2]=s2;
        }
    }
}

// Iter-2 winner: 32x8 NVEC=4 (128x8 tile, no Y-coarsening)
#define THX  32
#define THY   8
#define NVEC  4
#define OUT_W (THX * NVEC)  // 128
#define OUT_H THY            // 8
#define IN_W  (OUT_W + 2)    // 130
#define IN_H  (OUT_H + 2)    // 10

__global__ void depthwise_conv2d_3x3_nvec4(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int OH, int OW
) {
    __shared__ float sdata[IN_H][IN_W + 2];

    int tx = threadIdx.x, ty = threadIdx.y;
    int ow_base = blockIdx.x * OUT_W;
    int oh_base = blockIdx.y * OUT_H;
    int nc = blockIdx.z, n = nc / C, c = nc % C;

    const float* wp = weight + c * 9;
    float w00=wp[0],w01=wp[1],w02=wp[2],w10=wp[3],w11=wp[4],w12=wp[5],w20=wp[6],w21=wp[7],w22=wp[8];
    const float* inp = input + (n * C + c) * (H * W);

    int flat_tid = ty * THX + tx;
    #pragma unroll 6
    for (int i = flat_tid; i < IN_H * IN_W; i += THX * THY) {
        int si = i / IN_W, sj = i % IN_W;
        int ih = oh_base + si, iw = ow_base + sj;
        float val = ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                    ? __ldg(&inp[ih * W + iw]) : 0.f;
        sdata[si][sj] = val;
    }
    __syncthreads();

    int oh = oh_base + ty, ow0 = ow_base + tx * NVEC;
    if (oh < OH) {
        int sy = ty, sx = tx * NVEC;
        float s0=0,s1=0,s2=0,s3=0;

        float r0_0=sdata[sy  ][sx  ],r0_1=sdata[sy  ][sx+1],r0_2=sdata[sy  ][sx+2];
        float r0_3=sdata[sy  ][sx+3],r0_4=sdata[sy  ][sx+4],r0_5=sdata[sy  ][sx+5];
        s0+=w00*r0_0+w01*r0_1+w02*r0_2; s1+=w00*r0_1+w01*r0_2+w02*r0_3;
        s2+=w00*r0_2+w01*r0_3+w02*r0_4; s3+=w00*r0_3+w01*r0_4+w02*r0_5;

        float r1_0=sdata[sy+1][sx  ],r1_1=sdata[sy+1][sx+1],r1_2=sdata[sy+1][sx+2];
        float r1_3=sdata[sy+1][sx+3],r1_4=sdata[sy+1][sx+4],r1_5=sdata[sy+1][sx+5];
        s0+=w10*r1_0+w11*r1_1+w12*r1_2; s1+=w10*r1_1+w11*r1_2+w12*r1_3;
        s2+=w10*r1_2+w11*r1_3+w12*r1_4; s3+=w10*r1_3+w11*r1_4+w12*r1_5;

        float r2_0=sdata[sy+2][sx  ],r2_1=sdata[sy+2][sx+1],r2_2=sdata[sy+2][sx+2];
        float r2_3=sdata[sy+2][sx+3],r2_4=sdata[sy+2][sx+4],r2_5=sdata[sy+2][sx+5];
        s0+=w20*r2_0+w21*r2_1+w22*r2_2; s1+=w20*r2_1+w21*r2_2+w22*r2_3;
        s2+=w20*r2_2+w21*r2_3+w22*r2_4; s3+=w20*r2_3+w21*r2_4+w22*r2_5;

        if (bias != nullptr) { float b=bias[c]; s0+=b;s1+=b;s2+=b;s3+=b; }
        float* outp = output + ((n*C+c)*OH+oh)*OW + ow0;
        int rem = OW - ow0;
        if (rem >= 4) { outp[0]=s0;outp[1]=s1;outp[2]=s2;outp[3]=s3; }
        else { if(rem>0)outp[0]=s0; if(rem>1)outp[1]=s1; if(rem>2)outp[2]=s2; }
    }
}

// General stride/pad 3x3
#define GEN_TW 32
#define GEN_TH 8
__global__ void depthwise_conv2d_3x3_general(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int N,int C,int H,int W,int OH,int OW,int stride_h,int stride_w,int pad_h,int pad_w
) {
    __shared__ float sdata_gen[GEN_TH+2][GEN_TW+4];
    int tx=threadIdx.x,ty=threadIdx.y;
    int ow_base=blockIdx.x*GEN_TW,oh_base=blockIdx.y*GEN_TH;
    int nc=blockIdx.z,n=nc/C,c=nc%C;
    const float* wp=weight+c*9;
    float w00=wp[0],w01=wp[1],w02=wp[2],w10=wp[3],w11=wp[4],w12=wp[5],w20=wp[6],w21=wp[7],w22=wp[8];
    const float* inp=input+(n*C+c)*(H*W);
    int flat_tid=ty*GEN_TW+tx;
    for(int i=flat_tid;i<(GEN_TH+2)*(GEN_TW+2);i+=GEN_TW*GEN_TH){
        int si=i/(GEN_TW+2),sj=i%(GEN_TW+2);
        int ih=oh_base*stride_h-pad_h+si,iw=ow_base*stride_w-pad_w+sj;
        sdata_gen[si][sj]=((unsigned)ih<(unsigned)H&&(unsigned)iw<(unsigned)W)?inp[ih*W+iw]:0.f;
    }
    __syncthreads();
    int oh=oh_base+ty,ow=ow_base+tx;
    if(oh<OH&&ow<OW){
        int sy=ty*stride_h,sx=tx*stride_w;
        float sum=w00*sdata_gen[sy][sx]+w01*sdata_gen[sy][sx+1]+w02*sdata_gen[sy][sx+2]
                 +w10*sdata_gen[sy+1][sx]+w11*sdata_gen[sy+1][sx+1]+w12*sdata_gen[sy+1][sx+2]
                 +w20*sdata_gen[sy+2][sx]+w21*sdata_gen[sy+2][sx+1]+w22*sdata_gen[sy+2][sx+2];
        if(bias)sum+=bias[c];
        output[((n*C+c)*OH+oh)*OW+ow]=sum;
    }
}

__global__ void depthwise_conv2d_general_kernel(
    const float* __restrict__ input, const float* __restrict__ weight,
    const float* __restrict__ bias, float* __restrict__ output,
    int N,int C,int H,int W,int OH,int OW,int KH,int KW,int stride_h,int stride_w,int pad_h,int pad_w
) {
    int ow=blockIdx.x*blockDim.x+threadIdx.x,oh=blockIdx.y*blockDim.y+threadIdx.y;
    int nc=blockIdx.z,n=nc/C,c=nc%C;
    if(ow>=OW||oh>=OH)return;
    const float* wp=weight+c*KH*KW; const float* inp=input+(n*C+c)*H*W;
    float sum=0.f;
    for(int kh=0;kh<KH;kh++){int ih=oh*stride_h-pad_h+kh;if((unsigned)ih>=(unsigned)H)continue;
        for(int kw=0;kw<KW;kw++){int iw=ow*stride_w-pad_w+kw;if((unsigned)iw>=(unsigned)W)continue;
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
        // Choose between NVEC_Y=2 and NVEC_Y=1
        // NVEC_Y=2 kernel:
        dim3 block_a(THX_A, THY_A);
        dim3 grid_a((OW+OUT_W_A-1)/OUT_W_A,(OH+OUT_H_A-1)/OUT_H_A,N*C);
        depthwise_conv2d_3x3_coarse_y<<<grid_a,block_a>>>(
            input.data_ptr<float>(),w.data_ptr<float>(),bp,out.data_ptr<float>(),N,C,H,W,OH,OW);
    } else if(KH==3&&KW==3){
        dim3 block(GEN_TW,GEN_TH);
        dim3 grid((OW+GEN_TW-1)/GEN_TW,(OH+GEN_TH-1)/GEN_TH,N*C);
        depthwise_conv2d_3x3_general<<<grid,block>>>(
            input.data_ptr<float>(),w.data_ptr<float>(),bp,out.data_ptr<float>(),N,C,H,W,OH,OW,sh,sw,ph,pw);
    } else {
        dim3 block(32,8);dim3 grid((OW+31)/32,(OH+7)/8,N*C);
        depthwise_conv2d_general_kernel<<<grid,block>>>(
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
    name="depthwise_conv2d_ext_v7",
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
