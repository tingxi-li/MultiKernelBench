import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Depthwise conv2d optimized CUDA kernel - Iter 3
# Strategy: Process 2 NC-planes per block simultaneously
#   - Block handles nc=blockIdx.z*2 and nc=blockIdx.z*2+1
#   - Each thread computes NVEC=8 outputs for BOTH channels
#   - Halves grid size in Z: N*C/2 blocks instead of N*C
#   - Can overlap memory loads for two channels: thread loads for ch0 while
#     computing for ch1 → hide global memory latency
#   - 256 threads: 32 × 8 → 8 outputs × 2 channels = 16 outputs per thread
#   - smem: 2 channels × 260×10 = 5200 floats = 20.8KB (fits easily)

_depthwise_conv_src = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// ===========================================================================
// Iter 3: Process 2 channels per block (dual-channel fusion)
// smem: 2 x (10 rows x 260 cols) = 5200 floats = 20.8KB
// Each thread: NVEC=8 outputs x 2 channels = 16 total
// Grid: (OW+255)/256, (OH+7)/8, N*C/2 (if C even, else handle edge)
// ===========================================================================
#define T3_THX   32
#define T3_THY    8
#define T3_NVEC   8
#define T3_OUT_W (T3_THX * T3_NVEC)   // 256
#define T3_OUT_H  T3_THY               // 8
#define T3_IN_W  (T3_OUT_W + 2)        // 258
#define T3_IN_H  (T3_OUT_H + 2)        // 10

__global__ void depthwise_conv2d_3x3_v3(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W, int OH, int OW
) {
    // smem for two channels: ch0=[0..9][0..259], ch1=[0..9][0..259]
    __shared__ float s0[T3_IN_H][T3_IN_W + 2];  // 10 x 260 for channel 0
    __shared__ float s1[T3_IN_H][T3_IN_W + 2];  // 10 x 260 for channel 1

    int tx = threadIdx.x, ty = threadIdx.y;
    int ow_base = blockIdx.x * T3_OUT_W;
    int oh_base = blockIdx.y * T3_OUT_H;
    int nc2 = blockIdx.z;   // each block handles nc2*2 and nc2*2+1
    int nc0 = nc2 * 2;
    int nc1 = nc0 + 1;

    int n0  = nc0 / C, c0 = nc0 % C;
    int n1  = nc1 / C, c1 = nc1 % C;

    const float* inp0 = input + (n0 * C + c0) * (H * W);
    const float* inp1 = input + (n1 * C + c1) * (H * W);

    // Load weights for both channels
    const float* wp0 = weight + c0 * 9;
    float a00=wp0[0],a01=wp0[1],a02=wp0[2],a10=wp0[3],a11=wp0[4],a12=wp0[5],a20=wp0[6],a21=wp0[7],a22=wp0[8];
    const float* wp1 = weight + c1 * 9;
    float b00=wp1[0],b01=wp1[1],b02=wp1[2],b10=wp1[3],b11=wp1[4],b12=wp1[5],b20=wp1[6],b21=wp1[7],b22=wp1[8];

    int flat_tid = ty * T3_THX + tx;  // 0..255

    // Load smem for both channels simultaneously using float4
    // 10 rows x 64 float4 = 640 float4 per channel; 256 threads → 2.5 each
    // Total: 1280 float4 loads for 2 channels
    #pragma unroll 5
    for (int i = flat_tid; i < T3_IN_H * 64; i += T3_THX * T3_THY) {
        int row  = i / 64;
        int col4 = i % 64;
        int ih = oh_base + row;
        int iw = ow_base + col4 * 4;

        float4 v0 = make_float4(0.f,0.f,0.f,0.f);
        float4 v1 = make_float4(0.f,0.f,0.f,0.f);
        if ((unsigned)ih < (unsigned)H) {
            if ((unsigned)(iw + 3) < (unsigned)W) {
                v0 = __ldg((const float4*)(inp0 + ih * W + iw));
                v1 = __ldg((const float4*)(inp1 + ih * W + iw));
            } else {
                for (int k = 0; k < 4 && iw+k < W; k++) {
                    ((float*)&v0)[k] = __ldg(&inp0[ih*W+iw+k]);
                    ((float*)&v1)[k] = __ldg(&inp1[ih*W+iw+k]);
                }
            }
        }
        s0[row][col4*4  ] = v0.x; s0[row][col4*4+1] = v0.y;
        s0[row][col4*4+2] = v0.z; s0[row][col4*4+3] = v0.w;
        s1[row][col4*4  ] = v1.x; s1[row][col4*4+1] = v1.y;
        s1[row][col4*4+2] = v1.z; s1[row][col4*4+3] = v1.w;
    }

    // Trailing 2 cols (256, 257) for both channels: 10*2*2 = 40 scalar loads
    if (flat_tid < T3_IN_H * 2) {
        int row = flat_tid / 2;
        int col = 256 + (flat_tid & 1);
        int ih  = oh_base + row;
        int iw  = ow_base + col;
        int valid = ((unsigned)ih < (unsigned)H && (unsigned)iw < (unsigned)W);
        s0[row][col] = valid ? __ldg(&inp0[ih*W+iw]) : 0.f;
        s1[row][col] = valid ? __ldg(&inp1[ih*W+iw]) : 0.f;
    }

    __syncthreads();

    int oh = oh_base + ty;
    if (oh >= OH) return;

    int sx  = tx * T3_NVEC;
    int ow0 = ow_base + sx;
    int rem = OW - ow0;

    // Read smem rows for channel 0
    float r0_0=s0[ty  ][sx  ],r0_1=s0[ty  ][sx+1],r0_2=s0[ty  ][sx+2];
    float r0_3=s0[ty  ][sx+3],r0_4=s0[ty  ][sx+4],r0_5=s0[ty  ][sx+5];
    float r0_6=s0[ty  ][sx+6],r0_7=s0[ty  ][sx+7],r0_8=s0[ty  ][sx+8],r0_9=s0[ty  ][sx+9];
    float r1_0=s0[ty+1][sx  ],r1_1=s0[ty+1][sx+1],r1_2=s0[ty+1][sx+2];
    float r1_3=s0[ty+1][sx+3],r1_4=s0[ty+1][sx+4],r1_5=s0[ty+1][sx+5];
    float r1_6=s0[ty+1][sx+6],r1_7=s0[ty+1][sx+7],r1_8=s0[ty+1][sx+8],r1_9=s0[ty+1][sx+9];
    float r2_0=s0[ty+2][sx  ],r2_1=s0[ty+2][sx+1],r2_2=s0[ty+2][sx+2];
    float r2_3=s0[ty+2][sx+3],r2_4=s0[ty+2][sx+4],r2_5=s0[ty+2][sx+5];
    float r2_6=s0[ty+2][sx+6],r2_7=s0[ty+2][sx+7],r2_8=s0[ty+2][sx+8],r2_9=s0[ty+2][sx+9];

    // Compute 8 outputs for channel 0
    float p0=a00*r0_0+a01*r0_1+a02*r0_2+a10*r1_0+a11*r1_1+a12*r1_2+a20*r2_0+a21*r2_1+a22*r2_2;
    float p1=a00*r0_1+a01*r0_2+a02*r0_3+a10*r1_1+a11*r1_2+a12*r1_3+a20*r2_1+a21*r2_2+a22*r2_3;
    float p2=a00*r0_2+a01*r0_3+a02*r0_4+a10*r1_2+a11*r1_3+a12*r1_4+a20*r2_2+a21*r2_3+a22*r2_4;
    float p3=a00*r0_3+a01*r0_4+a02*r0_5+a10*r1_3+a11*r1_4+a12*r1_5+a20*r2_3+a21*r2_4+a22*r2_5;
    float p4=a00*r0_4+a01*r0_5+a02*r0_6+a10*r1_4+a11*r1_5+a12*r1_6+a20*r2_4+a21*r2_5+a22*r2_6;
    float p5=a00*r0_5+a01*r0_6+a02*r0_7+a10*r1_5+a11*r1_6+a12*r1_7+a20*r2_5+a21*r2_6+a22*r2_7;
    float p6=a00*r0_6+a01*r0_7+a02*r0_8+a10*r1_6+a11*r1_7+a12*r1_8+a20*r2_6+a21*r2_7+a22*r2_8;
    float p7=a00*r0_7+a01*r0_8+a02*r0_9+a10*r1_7+a11*r1_8+a12*r1_9+a20*r2_7+a21*r2_8+a22*r2_9;
    if (bias) { float ba=bias[c0]; p0+=ba;p1+=ba;p2+=ba;p3+=ba;p4+=ba;p5+=ba;p6+=ba;p7+=ba; }

    // Write channel 0 outputs
    float* outp0 = output + ((n0*C+c0)*OH+oh)*OW + ow0;
    if (rem >= 8) {
        outp0[0]=p0;outp0[1]=p1;outp0[2]=p2;outp0[3]=p3;
        outp0[4]=p4;outp0[5]=p5;outp0[6]=p6;outp0[7]=p7;
    } else {
        if(rem>0)outp0[0]=p0;if(rem>1)outp0[1]=p1;if(rem>2)outp0[2]=p2;
        if(rem>3)outp0[3]=p3;if(rem>4)outp0[4]=p4;if(rem>5)outp0[5]=p5;
        if(rem>6)outp0[6]=p6;
    }

    // Read smem rows for channel 1
    r0_0=s1[ty  ][sx  ];r0_1=s1[ty  ][sx+1];r0_2=s1[ty  ][sx+2];
    r0_3=s1[ty  ][sx+3];r0_4=s1[ty  ][sx+4];r0_5=s1[ty  ][sx+5];
    r0_6=s1[ty  ][sx+6];r0_7=s1[ty  ][sx+7];r0_8=s1[ty  ][sx+8];r0_9=s1[ty  ][sx+9];
    r1_0=s1[ty+1][sx  ];r1_1=s1[ty+1][sx+1];r1_2=s1[ty+1][sx+2];
    r1_3=s1[ty+1][sx+3];r1_4=s1[ty+1][sx+4];r1_5=s1[ty+1][sx+5];
    r1_6=s1[ty+1][sx+6];r1_7=s1[ty+1][sx+7];r1_8=s1[ty+1][sx+8];r1_9=s1[ty+1][sx+9];
    r2_0=s1[ty+2][sx  ];r2_1=s1[ty+2][sx+1];r2_2=s1[ty+2][sx+2];
    r2_3=s1[ty+2][sx+3];r2_4=s1[ty+2][sx+4];r2_5=s1[ty+2][sx+5];
    r2_6=s1[ty+2][sx+6];r2_7=s1[ty+2][sx+7];r2_8=s1[ty+2][sx+8];r2_9=s1[ty+2][sx+9];

    // Compute 8 outputs for channel 1
    float q0=b00*r0_0+b01*r0_1+b02*r0_2+b10*r1_0+b11*r1_1+b12*r1_2+b20*r2_0+b21*r2_1+b22*r2_2;
    float q1=b00*r0_1+b01*r0_2+b02*r0_3+b10*r1_1+b11*r1_2+b12*r1_3+b20*r2_1+b21*r2_2+b22*r2_3;
    float q2=b00*r0_2+b01*r0_3+b02*r0_4+b10*r1_2+b11*r1_3+b12*r1_4+b20*r2_2+b21*r2_3+b22*r2_4;
    float q3=b00*r0_3+b01*r0_4+b02*r0_5+b10*r1_3+b11*r1_4+b12*r1_5+b20*r2_3+b21*r2_4+b22*r2_5;
    float q4=b00*r0_4+b01*r0_5+b02*r0_6+b10*r1_4+b11*r1_5+b12*r1_6+b20*r2_4+b21*r2_5+b22*r2_6;
    float q5=b00*r0_5+b01*r0_6+b02*r0_7+b10*r1_5+b11*r1_6+b12*r1_7+b20*r2_5+b21*r2_6+b22*r2_7;
    float q6=b00*r0_6+b01*r0_7+b02*r0_8+b10*r1_6+b11*r1_7+b12*r1_8+b20*r2_6+b21*r2_7+b22*r2_8;
    float q7=b00*r0_7+b01*r0_8+b02*r0_9+b10*r1_7+b11*r1_8+b12*r1_9+b20*r2_7+b21*r2_8+b22*r2_9;
    if (bias) { float bb=bias[c1]; q0+=bb;q1+=bb;q2+=bb;q3+=bb;q4+=bb;q5+=bb;q6+=bb;q7+=bb; }

    // Write channel 1 outputs
    float* outp1 = output + ((n1*C+c1)*OH+oh)*OW + ow0;
    if (rem >= 8) {
        outp1[0]=q0;outp1[1]=q1;outp1[2]=q2;outp1[3]=q3;
        outp1[4]=q4;outp1[5]=q5;outp1[6]=q6;outp1[7]=q7;
    } else {
        if(rem>0)outp1[0]=q0;if(rem>1)outp1[1]=q1;if(rem>2)outp1[2]=q2;
        if(rem>3)outp1[3]=q3;if(rem>4)outp1[4]=q4;if(rem>5)outp1[5]=q5;
        if(rem>6)outp1[6]=q6;
    }
}

// ===========================================================================
// Best prior (iter-1): NVEC=8, 256x8 tile, float4 smem loads, scalar outputs
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
        // Iter 3: dual-channel fusion (2 NC-planes per block)
        // Grid: (OW+255)/256 x (OH+7)/8 x N*C/2
        int nc_total = N * C;
        if (nc_total % 2 == 0) {
            dim3 block(T3_THX, T3_THY);
            dim3 grid((OW+T3_OUT_W-1)/T3_OUT_W, (OH+T3_OUT_H-1)/T3_OUT_H, nc_total/2);
            depthwise_conv2d_3x3_v3<<<grid,block>>>(
                input.data_ptr<float>(),w.data_ptr<float>(),bp,out.data_ptr<float>(),N,C,H,W,OH,OW);
        } else {
            // fallback: single channel kernel
            dim3 block(T1_THX, T1_THY);
            dim3 grid((OW+T1_OUT_W-1)/T1_OUT_W, (OH+T1_OUT_H-1)/T1_OUT_H, nc_total);
            depthwise_conv2d_3x3_v1<<<grid,block>>>(
                input.data_ptr<float>(),w.data_ptr<float>(),bp,out.data_ptr<float>(),N,C,H,W,OH,OW);
        }
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
    name="depthwise_conv2d_ext_v13",
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
