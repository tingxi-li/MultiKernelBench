#include <torch/extension.h>
#include <torch/extension.h>
torch::Tensor gemm(torch::Tensor A, torch::Tensor B);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("gemm", torch::wrap_pybind_function(gemm), "gemm");
}