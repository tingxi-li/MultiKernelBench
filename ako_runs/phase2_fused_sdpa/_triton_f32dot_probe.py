import torch, triton, triton.language as tl

@triton.jit
def kd(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, IP: tl.constexpr):
    om = tl.arange(0, M); on = tl.arange(0, N); ok = tl.arange(0, K)
    a = tl.load(A + om[:, None]*K + ok[None, :])
    b = tl.load(B + ok[:, None]*N + on[None, :])
    if IP == 0:
        c = tl.dot(a, b, input_precision="tf32")
    elif IP == 1:
        c = tl.dot(a, b, input_precision="ieee")
    elif IP == 2:
        c = tl.dot(a, b, input_precision="tf32x3")
    else:
        c = tl.dot(a.to(tl.float16), b.to(tl.float16))
    tl.store(C + om[:, None]*N + on[None, :], c)

M=N=K=64
torch.manual_seed(0)
a = torch.rand(M,K,device='cuda'); b = torch.rand(K,N,device='cuda')
ref = (a.double()@b.double())
for ip,name in [(0,'tf32'),(1,'ieee'),(2,'tf32x3'),(3,'fp16')]:
    c = torch.empty(M,N,device='cuda')
    kd[(1,)](a,b,c,M=M,N=N,K=K,IP=ip,num_warps=4)
    e = (c.double()-ref)
    print(f"{name:7s} max_abs={e.abs().max():.4e} signed_mean={e.mean():.4e} rel_mean={(e/ref).mean():.4e}")
