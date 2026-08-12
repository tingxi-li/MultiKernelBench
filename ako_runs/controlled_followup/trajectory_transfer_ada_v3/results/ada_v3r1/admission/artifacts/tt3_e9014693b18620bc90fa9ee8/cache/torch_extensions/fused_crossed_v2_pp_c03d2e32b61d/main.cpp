#include <torch/extension.h>
#include <torch/extension.h>
torch::Tensor common_postprocess(torch::Tensor input, torch::Tensor bias);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("common_postprocess", torch::wrap_pybind_function(common_postprocess), "common_postprocess");
}