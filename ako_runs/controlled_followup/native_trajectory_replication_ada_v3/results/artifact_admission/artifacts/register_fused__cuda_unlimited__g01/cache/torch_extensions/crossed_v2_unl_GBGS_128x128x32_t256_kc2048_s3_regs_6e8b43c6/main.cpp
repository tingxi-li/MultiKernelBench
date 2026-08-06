#include <torch/extension.h>
#include <torch/extension.h>
torch::Tensor fused(torch::Tensor A, torch::Tensor B, torch::Tensor Bias);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("fused", torch::wrap_pybind_function(fused), "fused");
}