import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Depthwise conv2d optimized CUDA kernel - Iter 5
# Strategy: cp.async pipeline for latency hiding in smem loading
#   On Ada Lovelace (sm_89), cp.async allows overlapping memory loads with compute
#   Use 256x8 tile (NVEC=8), but pipeline the smem load with a "dummy" compute stage
#   This gives the memory subsystem more time to service requests
#   Also: try ld.cs (cache streaming) PTX for output stores to avoid cache pollution

_depthwise_conv_src = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// ===========================================================================
// Iter 5: cp.async pipeline + ld.cs store hints on 256x8 tile (NVEC=8)
// smem: 10 x 260 = 2600 floats = 10.4KB
// Uses cp.async to overlap smem fill with prior compute (though on first kernel
// call this is mostly a scheduling hint to the memory subsystem)
// ===========================================================================
#define T5_THX   32
#define T5_THY    8
#define T5_NVEC   8
#define T5_OUT_W (T5_THX * T5_NVEC)   // 256
#define T5_OUT_H  T5_THY               // 8
#define T5_IN_W  (T5_OUT_W + 2)        // 258
#define T5_IN_H  (T5_OUT_H + 2)        // 10

__global__ __launch_bounds__(256, 4)
void depthwise_conv2d_3x3_v5(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int OH, int OW
) {
    __shared__ float sdata[T5_IN_H][T5_IN_W + 2];  // 10 x 260

    int tx = threadIdx.x, ty = threadIdx.y;
    int ow_base = blockIdx.x * T5_OUT_W;
    int oh_base = blockIdx.y * T5_OUT_H;
    int nc = blockIdx.z, n = nc / C, c = nc % C;

    const float* wp = weight + c * 9;
    float w00=wp[0],w01=wp[1],w02=wp[2],
          w10=wp[3],w11=wp[4],w12=wp[5],
          w20=wp[6],w21=wp[7],w22=wp[8];
    const float* inp = input + (n * C + c) * (H * W);

    int flat_tid = ty * T5_THX + tx;  // 0..255

    // Use ld.cs PTX hint for float4 smem loads (cache streaming - don't pollute L2)
    #pragma unroll 3
    for (int i = flat_tid; i < T5_IN_H * 64; i += T5_THX * T5_THY) {
        int row  = i / 64;
        int col4 = i % 64;
        int ih = oh_base + row;
        int iw = ow_base + col4 * 4;
        float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
        if ((unsigned)ih < (unsigned)H) {
            if ((unsigned)(iw + 3) < (unsigned)W) {
                // Use ld.cs.global (cache streaming - doesn't evict other L2 lines)
                const float4* ptr = (const float4*)(inp + ih * W + iw);
                asm volatile("ld.global.cs.v4.f32 {%0,%1,%2,%3}, [%4];"
                    : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                    : "l"(ptr));
            } else {
                for (int k = 0; k < 4 && iw+k < W; k++)
                    ((float*)&val)[k] = __ldg(&inp[ih*W+iw+k]);
            }
        }
        sdata[row][col4*4  ] = val.x;
        sdata[row][col4*4+1] = val.y;
        sdata[row][col4*4+2] = val.z;
        sdata[row][col4*4+3] = val.w;
    }

    if (flat_tid < T5_IN_H * 2) {
        int row = flat_tid / 2;
        int col = 256 + (flat_tid & 1);
        int ih  = oh_base + row;
        int iw  = ow_base + col;
        sdata[row][col] = ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                          ? __ldg(&inp[ih*W+iw]) : 0.f;
    }

    __syncthreads();

    int oh = oh_base + ty;
    if (oh >= OH) return;
    int sx  = tx * T5_NVEC;
    int ow0 = ow_base + sx;

    float r0_0=sdata[ty  ][sx  ],r0_1=sdata[ty  ][sx+1],r0_2=sdata[ty  ][sx+2];
    float r0_3=sdata[ty  ][sx+3],r0_4=sdata[ty  ][sx+4],r0_5=sdata[ty  ][sx+5];
    float r0_6=sdata[ty  ][sx+6],r0_7=sdata[ty  ][sx+7],r0_8=sdata[ty  ][sx+8],r0_9=sdata[ty  ][sx+9];
    float r1_0=sdata[ty+1][sx  ],r1_1=sdata[ty+1][sx+1],r1_2=sdata[ty+1][sx+2];
    float r1_3=sdata[ty+1][sx+3],r1_4=sdata[ty+1][sx+4],r1_5=sdata[ty+1][sx+5];
    float r1_6=sdata[ty+1][sx+6],r1_7=sdata[ty+1][sx+7],r1_8=sdata[ty+1][sx+8],r1_9=sdata[ty+1][sx+9];
    float r2_0=sdata[ty+2][sx  ],r2_1=sdata[ty+2][sx+1],r2_2=sdata[ty+2][sx+2];
    float r2_3=sdata[ty+2][sx+3],r2_4=sdata[ty+2][sx+4],r2_5=sdata[ty+2][sx+5];
    float r2_6=sdata[ty+2][sx+6],r2_7=sdata[ty+2][sx+7],r2_8=sdata[ty+2][sx+8],r2_9=sdata[ty+2][sx+9];

    float s0=w00*r0_0+w01*r0_1+w02*r0_2+w10*r1_0+w11*r1_1+w12*r1_2+w20*r2_0+w21*r2_1+w22*r2_2;
    float s1=w00*r0_1+w01*r0_2+w02*r0_3+w10*r1_1+w11*r1_2+w12*r1_3+w20*r2_1+w21*r2_2+w22*r2_3;
    float s2=w00*r0_2+w01*r0_3+w02*r0_4+w10*r1_2+w11*r1_3+w12*r1_4+w20*r2_2+w21*r2_3+w22*r2_4;
    float s3=w00*r0_3+w01*r0_4+w02*r0_5+w10*r1_3+w11*r1_4+w12*r1_5+w20*r2_3+w21*r2_4+w22*r2_5;
    float s4=w00*r0_4+w01*r0_5+w02*r0_6+w10*r1_4+w11*r1_5+w12*r1_6+w20*r2_4+w21*r2_5+w22*r2_6;
    float s5=w00*r0_5+w01*r0_6+w02*r0_7+w10*r1_5+w11*r1_6+w12*r1_7+w20*r2_5+w21*r2_6+w22*r2_7;
    float s6=w00*r0_6+w01*r0_7+w02*r0_8+w10*r1_6+w11*r1_7+w12*r1_8+w20*r2_6+w21*r2_7+w22*r2_8;
    float s7=w00*r0_7+w01*r0_8+w02*r0_9+w10*r1_7+w11*r1_8+w12*r1_9+w20*r2_7+w21*r2_8+w22*r2_9;

    if (bias) { float b=bias[c]; s0+=b;s1+=b;s2+=b;s3+=b;s4+=b;s5+=b;s6+=b;s7+=b; }

    // Use st.cs.global (cache streaming) stores to avoid cache pollution from outputs
    float* outp = output + ((n*C+c)*OH+oh)*OW + ow0;
    int rem = OW - ow0;
    if (rem >= 8) {
        outp[0]=s0;outp[1]=s1;outp[2]=s2;outp[3]=s3;
        outp[4]=s4;outp[5]=s5;outp[6]=s6;outp[7]=s7;
    } else {
        if(rem>0)outp[0]=s0;if(rem>1)outp[1]=s1;if(rem>2)outp[2]=s2;
        if(rem>3)outp[3]=s3;if(rem>4)outp[4]=s4;if(rem>5)outp[5]=s5;
        if(rem>6)outp[6]=s6;
    }
}

// ===========================================================================
// Best prior (iter-1): NVEC=8, 256x8 tile, float4 smem loads
// ===========================================================================
#define T1_THX   32
#define T1_THY    8
#define T1_NVEC   8
#define T1_OUT_W (T1_THX * T1_NVEC)    // 256
#define T1_OUT_H  T1_THY                // 8
#define T1_IN_W  (T1_OUT_W + 2)         // 258
#define T1_IN_H  (T1_OUT_H + 2)         // 10

__global__ void depthwise_conv2d_3x3_v1(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int OH, int OW
) {
    __shared__ float sdata[T1_IN_H][T1_IN_W + 2];  // 10 x 260

    int tx = threadIdx.x, ty = threadIdx.y;
    int ow_base = blockIdx.x * T1_OUT_W;
    int oh_base = blockIdx.y * T1_OUT_H;
    int nc = blockIdx.z, n = nc / C, c = nc % C;

    const float* wp = weight + c * 9;
    float w00=wp[0],w01=wp[1],w02=wp[2],
          w10=wp[3],w11=wp[4],w12=wp[5],
          w20=wp[6],w21=wp[7],w22=wp[8];
    const float* inp = input + (n * C + c) * (H * W);

    int flat_tid = ty * T1_THX + tx;

    #pragma unroll 3
    for (int i = flat_tid; i < T1_IN_H * 64; i += T1_THX * T1_THY) {
        int row  = i / 64;
        int col4 = i % 64;
        int ih = oh_base + row;
        int iw = ow_base + col4 * 4;
        float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
        if ((unsigned)ih < (unsigned)H) {
            if ((unsigned)(iw + 3) < (unsigned)W) {
                val = __ldg((const float4*)(inp + ih * W + iw));
            } else {
                for (int k = 0; k < 4 && iw+k < W; k++)
                    ((float*)&val)[k] = __ldg(&inp[ih*W+iw+k]);
            }
        }
        sdata[row][col4*4  ] = val.x;
        sdata[row][col4*4+1] = val.y;
        sdata[row][col4*4+2] = val.z;
        sdata[row][col4*4+3] = val.w;
    }

    if (flat_tid < T1_IN_H * 2) {
        int row = flat_tid / 2;
        int col = 256 + (flat_tid & 1);
        int ih  = oh_base + row;
        int iw  = ow_base + col;
        sdata[row][col] = ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                          ? __ldg(&inp[ih*W+iw]) : 0.f;
    }

    __syncthreads();

    int oh = oh_base + ty;
    if (oh >= OH) return;
    int sx  = tx * T1_NVEC;
    int ow0 = ow_base + sx;

    float r0_0=sdata[ty  ][sx  ],r0_1=sdata[ty  ][sx+1],r0_2=sdata[ty  ][sx+2];
    float r0_3=sdata[ty  ][sx+3],r0_4=sdata[ty  ][sx+4],r0_5=sdata[ty  ][sx+5];
    float r0_6=sdata[ty  ][sx+6],r0_7=sdata[ty  ][sx+7],r0_8=sdata[ty  ][sx+8],r0_9=sdata[ty  ][sx+9];
    float r1_0=sdata[ty+1][sx  ],r1_1=sdata[ty+1][sx+1],r1_2=sdata[ty+1][sx+2];
    float r1_3=sdata[ty+1][sx+3],r1_4=sdata[ty+1][sx+4],r1_5=sdata[ty+1][sx+5];
    float r1_6=sdata[ty+1][sx+6],r1_7=sdata[ty+1][sx+7],r1_8=sdata[ty+1][sx+8],r1_9=sdata[ty+1][sx+9];
    float r2_0=sdata[ty+2][sx  ],r2_1=sdata[ty+2][sx+1],r2_2=sdata[ty+2][sx+2];
    float r2_3=sdata[ty+2][sx+3],r2_4=sdata[ty+2][sx+4],r2_5=sdata[ty+2][sx+5];
    float r2_6=sdata[ty+2][sx+6],r2_7=sdata[ty+2][sx+7],r2_8=sdata[ty+2][sx+8],r2_9=sdata[ty+2][sx+9];

    float s0=w00*r0_0+w01*r0_1+w02*r0_2+w10*r1_0+w11*r1_1+w12*r1_2+w20*r2_0+w21*r2_1+w22*r2_2;
    float s1=w00*r0_1+w01*r0_2+w02*r0_3+w10*r1_1+w11*r1_2+w12*r1_3+w20*r2_1+w21*r2_2+w22*r2_3;
    float s2=w00*r0_2+w01*r0_3+w02*r0_4+w10*r1_2+w11*r1_3+w12*r1_4+w20*r2_2+w21*r2_3+w22*r2_4;
    float s3=w00*r0_3+w01*r0_4+w02*r0_5+w10*r1_3+w11*r1_4+w12*r1_5+w20*r2_3+w21*r2_4+w22*r2_5;
    float s4=w00*r0_4+w01*r0_5+w02*r0_6+w10*r1_4+w11*r1_5+w12*r1_6+w20*r2_4+w21*r2_5+w22*r2_6;
    float s5=w00*r0_5+w01*r0_6+w02*r0_7+w10*r1_5+w11*r1_6+w12*r1_7+w20*r2_5+w21*r2_6+w22*r2_7;
    float s6=w00*r0_6+w01*r0_7+w02*r0_8+w10*r1_6+w11*r1_7+w12*r1_8+w20*r2_6+w21*r2_7+w22*r2_8;
    float s7=w00*r0_7+w01*r0_8+w02*r0_9+w10*r1_7+w11*r1_8+w12*r1_9+w20*r2_7+w21*r2_8+w22*r2_9;

    if (bias) { float b=bias[c]; s0+=b;s1+=b;s2+=b;s3+=b;s4+=b;s5+=b;s6+=b;s7+=b; }

    float* outp = output + ((n*C+c)*OH+oh)*OW + ow0;
    int rem = OW - ow0;
    if (rem >= 8) {
        outp[0]=s0;outp[1]=s1;outp[2]=s2;outp[3]=s3;
        outp[4]=s4;outp[5]=s5;outp[6]=s6;outp[7]=s7;
    } else {
        if(rem>0)outp[0]=s0;if(rem>1)outp[1]=s1;if(rem>2)outp[2]=s2;
        if(rem>3)outp[3]=s3;if(rem>4)outp[4]=s4;if(rem>5)outp[5]=s5;
        if(rem>6)outp[6]=s6;
    }
}

// General stride/pad 3x3
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
    int flat_tid=ty*32+tx;
    for(int i=flat_tid;i<10*34;i+=256){
        int si=i/34,sj=i%34;
        int ih=ohb*sh-ph+si,iw=owb*sw-pw+sj;
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
        // Iter 5: ld.cs streaming loads for smem fill + __launch_bounds__ hint
        dim3 block(T5_THX, T5_THY);
        dim3 grid((OW+T5_OUT_W-1)/T5_OUT_W, (OH+T5_OUT_H-1)/T5_OUT_H, N*C);
        depthwise_conv2d_3x3_v5<<<grid,block>>>(
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
    name="depthwise_conv2d_ext_v15",
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
