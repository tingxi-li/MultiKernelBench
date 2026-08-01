# crossed-v2 measured Triton capability request
REQUEST = {"BK": 64, "BM": 128, "BN": 256, "operation": "allocate fp32 BMxBN user-managed shared accumulator tile", "required_semantics": ["store tl.dot accumulator", "barrier", "reload", "bias", "exact GELU"], "triton_api": "public triton.language"}
import triton.language as tl
shared = tl.alloc_shared((REQUEST['BM'], REQUEST['BN']), tl.float32)
# The probe succeeds only if the pinned public API resolves this explicit allocation.
