import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Depthwise conv2d optimized CUDA kernel - Iter 6
# Strategy: NVEC=16 (512-wide tile), single block per row covers the entire output width
#   - OW=510: ONE block in X (510 < 512) → only 1 block per (row, channel, batch)
#   - Grid: 1 x 64 x 1024 = 65536 blocks (vs 2 x 64 x 1024 = 131072 for NVEC=8)
#   - Each thread: 16 outputs → maximum ILP
#   - smem: 514 x 10 = 5140 floats = 20.6KB per block
#   - For OW=510: ow_base=0, block covers cols 0..511 (but OW=510 → rem check needed)
#   - Only 1 block needed in X dimension for OW=510 !

_depthwise_conv_src = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// ===========================================================================
// Iter 6: NVEC=16 (512x8 output tile), 256 threads
// For OW=510: single block covers entire width (510 < 512)
// smem: 10 x 516 (514+2 padding) = 5160 floats = 20.6KB
// ===========================================================================
#define T6_THX   32
#define T6_THY    8
#define T6_NVEC   16
#define T6_OUT_W (T6_THX * T6_NVEC)   // 512
#define T6_OUT_H  T6_THY               // 8
#define T6_IN_W  (T6_OUT_W + 2)        // 514
#define T6_IN_H  (T6_OUT_H + 2)        // 10

__global__ __launch_bounds__(256, 3)
void depthwise_conv2d_3x3_v6(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int OH, int OW
) {
    __shared__ float sdata[T6_IN_H][T6_IN_W + 2];  // 10 x 516, 20.6KB

    int tx = threadIdx.x, ty = threadIdx.y;
    int ow_base = blockIdx.x * T6_OUT_W;  // 0 for OW=510
    int oh_base = blockIdx.y * T6_OUT_H;
    int nc = blockIdx.z, n = nc / C, c = nc % C;

    const float* wp = weight + c * 9;
    float w00=wp[0],w01=wp[1],w02=wp[2],
          w10=wp[3],w11=wp[4],w12=wp[5],
          w20=wp[6],w21=wp[7],w22=wp[8];
    const float* inp = input + (n * C + c) * (H * W);

    int flat_tid = ty * T6_THX + tx;  // 0..255

    // Fill smem: 10 rows x 514 cols via float4 (128 float4 per row) + 2 scalar per row
    // 10 x 128 = 1280 float4 loads; 256 threads → 5 each
    #pragma unroll 5
    for (int i = flat_tid; i < T6_IN_H * 128; i += T6_THX * T6_THY) {
        int row  = i / 128;
        int col4 = i % 128;
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

    // Trailing 2 cols (512, 513) for each of 10 rows: 20 scalar loads
    if (flat_tid < T6_IN_H * 2) {
        int row = flat_tid / 2;
        int col = 512 + (flat_tid & 1);
        int ih  = oh_base + row;
        int iw  = ow_base + col;
        sdata[row][col] = ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W)
                          ? __ldg(&inp[ih*W+iw]) : 0.f;
    }

    __syncthreads();

    int oh = oh_base + ty;
    if (oh >= OH) return;

    int sx  = tx * T6_NVEC;  // 0,16,32,...,496
    int ow0 = ow_base + sx;

    // Load 3 rows x 18 smem values
    // Row 0
    float r0_0 =sdata[ty  ][sx   ],r0_1 =sdata[ty  ][sx+1 ],r0_2 =sdata[ty  ][sx+2 ];
    float r0_3 =sdata[ty  ][sx+3 ],r0_4 =sdata[ty  ][sx+4 ],r0_5 =sdata[ty  ][sx+5 ];
    float r0_6 =sdata[ty  ][sx+6 ],r0_7 =sdata[ty  ][sx+7 ],r0_8 =sdata[ty  ][sx+8 ];
    float r0_9 =sdata[ty  ][sx+9 ],r0_10=sdata[ty  ][sx+10],r0_11=sdata[ty  ][sx+11];
    float r0_12=sdata[ty  ][sx+12],r0_13=sdata[ty  ][sx+13],r0_14=sdata[ty  ][sx+14];
    float r0_15=sdata[ty  ][sx+15],r0_16=sdata[ty  ][sx+16],r0_17=sdata[ty  ][sx+17];
    // Row 1
    float r1_0 =sdata[ty+1][sx   ],r1_1 =sdata[ty+1][sx+1 ],r1_2 =sdata[ty+1][sx+2 ];
    float r1_3 =sdata[ty+1][sx+3 ],r1_4 =sdata[ty+1][sx+4 ],r1_5 =sdata[ty+1][sx+5 ];
    float r1_6 =sdata[ty+1][sx+6 ],r1_7 =sdata[ty+1][sx+7 ],r1_8 =sdata[ty+1][sx+8 ];
    float r1_9 =sdata[ty+1][sx+9 ],r1_10=sdata[ty+1][sx+10],r1_11=sdata[ty+1][sx+11];
    float r1_12=sdata[ty+1][sx+12],r1_13=sdata[ty+1][sx+13],r1_14=sdata[ty+1][sx+14];
    float r1_15=sdata[ty+1][sx+15],r1_16=sdata[ty+1][sx+16],r1_17=sdata[ty+1][sx+17];
    // Row 2
    float r2_0 =sdata[ty+2][sx   ],r2_1 =sdata[ty+2][sx+1 ],r2_2 =sdata[ty+2][sx+2 ];
    float r2_3 =sdata[ty+2][sx+3 ],r2_4 =sdata[ty+2][sx+4 ],r2_5 =sdata[ty+2][sx+5 ];
    float r2_6 =sdata[ty+2][sx+6 ],r2_7 =sdata[ty+2][sx+7 ],r2_8 =sdata[ty+2][sx+8 ];
    float r2_9 =sdata[ty+2][sx+9 ],r2_10=sdata[ty+2][sx+10],r2_11=sdata[ty+2][sx+11];
    float r2_12=sdata[ty+2][sx+12],r2_13=sdata[ty+2][sx+13],r2_14=sdata[ty+2][sx+14];
    float r2_15=sdata[ty+2][sx+15],r2_16=sdata[ty+2][sx+16],r2_17=sdata[ty+2][sx+17];

    // Compute 16 outputs
    float s0 =w00*r0_0 +w01*r0_1 +w02*r0_2 +w10*r1_0 +w11*r1_1 +w12*r1_2 +w20*r2_0 +w21*r2_1 +w22*r2_2;
    float s1 =w00*r0_1 +w01*r0_2 +w02*r0_3 +w10*r1_1 +w11*r1_2 +w12*r1_3 +w20*r2_1 +w21*r2_2 +w22*r2_3;
    float s2 =w00*r0_2 +w01*r0_3 +w02*r0_4 +w10*r1_2 +w11*r1_3 +w12*r1_4 +w20*r2_2 +w21*r2_3 +w22*r2_4;
    float s3 =w00*r0_3 +w01*r0_4 +w02*r0_5 +w10*r1_3 +w11*r1_4 +w12*r1_5 +w20*r2_3 +w21*r2_4 +w22*r2_5;
    float s4 =w00*r0_4 +w01*r0_5 +w02*r0_6 +w10*r1_4 +w11*r1_5 +w12*r1_6 +w20*r2_4 +w21*r2_5 +w22*r2_6;
    float s5 =w00*r0_5 +w01*r0_6 +w02*r0_7 +w10*r1_5 +w11*r1_6 +w12*r1_7 +w20*r2_5 +w21*r2_6 +w22*r2_7;
    float s6 =w00*r0_6 +w01*r0_7 +w02*r0_8 +w10*r1_6 +w11*r1_7 +w12*r1_8 +w20*r2_6 +w21*r2_7 +w22*r2_8;
    float s7 =w00*r0_7 +w01*r0_8 +w02*r0_9 +w10*r1_7 +w11*r1_8 +w12*r1_9 +w20*r2_7 +w21*r2_8 +w22*r2_9;
    float s8 =w00*r0_8 +w01*r0_9 +w02*r0_10+w10*r1_8 +w11*r1_9 +w12*r1_10+w20*r2_8 +w21*r2_9 +w22*r2_10;
    float s9 =w00*r0_9 +w01*r0_10+w02*r0_11+w10*r1_9 +w11*r1_10+w12*r1_11+w20*r2_9 +w21*r2_10+w22*r2_11;
    float s10=w00*r0_10+w01*r0_11+w02*r0_12+w10*r1_10+w11*r1_11+w12*r1_12+w20*r2_10+w21*r2_11+w22*r2_12;
    float s11=w00*r0_11+w01*r0_12+w02*r0_13+w10*r1_11+w11*r1_12+w12*r1_13+w20*r2_11+w21*r2_12+w22*r2_13;
    float s12=w00*r0_12+w01*r0_13+w02*r0_14+w10*r1_12+w11*r1_13+w12*r1_14+w20*r2_12+w21*r2_13+w22*r2_14;
    float s13=w00*r0_13+w01*r0_14+w02*r0_15+w10*r1_13+w11*r1_14+w12*r1_15+w20*r2_13+w21*r2_14+w22*r2_15;
    float s14=w00*r0_14+w01*r0_15+w02*r0_16+w10*r1_14+w11*r1_15+w12*r1_16+w20*r2_14+w21*r2_15+w22*r2_16;
    float s15=w00*r0_15+w01*r0_16+w02*r0_17+w10*r1_15+w11*r1_16+w12*r1_17+w20*r2_15+w21*r2_16+w22*r2_17;

    if (bias) {
        float b=bias[c];
        s0+=b;s1+=b;s2+=b;s3+=b;s4+=b;s5+=b;s6+=b;s7+=b;
        s8+=b;s9+=b;s10+=b;s11+=b;s12+=b;s13+=b;s14+=b;s15+=b;
    }

    float* outp = output + ((n*C+c)*OH+oh)*OW + ow0;
    int rem = OW - ow0;
    if (rem >= 16) {
        outp[0]=s0;outp[1]=s1;outp[2]=s2;outp[3]=s3;
        outp[4]=s4;outp[5]=s5;outp[6]=s6;outp[7]=s7;
        outp[8]=s8;outp[9]=s9;outp[10]=s10;outp[11]=s11;
        outp[12]=s12;outp[13]=s13;outp[14]=s14;outp[15]=s15;
    } else {
        if(rem>0)outp[0]=s0;if(rem>1)outp[1]=s1;if(rem>2)outp[2]=s2;
        if(rem>3)outp[3]=s3;if(rem>4)outp[4]=s4;if(rem>5)outp[5]=s5;
        if(rem>6)outp[6]=s6;if(rem>7)outp[7]=s7;if(rem>8)outp[8]=s8;
        if(rem>9)outp[9]=s9;if(rem>10)outp[10]=s10;if(rem>11)outp[11]=s11;
        if(rem>12)outp[12]=s12;if(rem>13)outp[13]=s13;if(rem>14)outp[14]=s14;
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
        // Iter 6: NVEC=16 (512-wide tile) → single block in X for OW=510
        dim3 block(T6_THX, T6_THY);
        dim3 grid((OW+T6_OUT_W-1)/T6_OUT_W, (OH+T6_OUT_H-1)/T6_OUT_H, N*C);
        depthwise_conv2d_3x3_v6<<<grid,block>>>(
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
    name="depthwise_conv2d_ext_v16",
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
