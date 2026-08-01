from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
HEADER = HERE / "checked_cuda_launch.h"


class CheckedCudaLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        compiler = shutil.which("c++") or shutil.which("g++")
        if compiler is None:
            raise unittest.SkipTest("no C++ compiler available")
        cls._temporary = tempfile.TemporaryDirectory()
        root = Path(cls._temporary.name)
        (root / "torch").mkdir()
        (root / "cuda_runtime_api.h").write_text(
            textwrap.dedent(
                """
                #pragma once
                using cudaError_t = int;
                constexpr cudaError_t cudaSuccess = 0;
                constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize = 1;
                extern cudaError_t injected_attribute_result;
                extern cudaError_t injected_launch_result;
                inline cudaError_t cudaFuncSetAttribute(const void*, int, int) {
                    return injected_attribute_result;
                }
                inline cudaError_t cudaGetLastError() {
                    return injected_launch_result;
                }
                inline const char* cudaGetErrorString(cudaError_t) {
                    return "injected CUDA error";
                }
                """
            ).lstrip(),
            encoding="utf-8",
        )
        (root / "torch" / "extension.h").write_text(
            textwrap.dedent(
                """
                #pragma once
                #include <stdexcept>
                #define TORCH_CHECK(condition, ...) do { \\
                    if (!(condition)) throw std::runtime_error("TORCH_CHECK"); \\
                } while (0)
                """
            ).lstrip(),
            encoding="utf-8",
        )
        harness = root / "harness.cpp"
        harness.write_text(
            textwrap.dedent(
                f"""
                #include <string>
                #include "{HEADER}"

                cudaError_t injected_attribute_result = cudaSuccess;
                cudaError_t injected_launch_result = cudaSuccess;

                int main(int argc, char** argv) {{
                    std::string mode = argc > 1 ? argv[1] : "success";
                    if (mode == "attribute_failure") {{
                        injected_attribute_result = 7;
                        try {{
                            checked_dynamic_smem(nullptr, 1024, "kernel");
                        }} catch (const std::runtime_error&) {{
                            return 0;
                        }}
                        return 10;
                    }}
                    if (mode == "launch_failure") {{
                        injected_launch_result = 9;
                        try {{
                            checked_kernel_launch("kernel");
                        }} catch (const std::runtime_error&) {{
                            return 0;
                        }}
                        return 11;
                    }}
                    checked_dynamic_smem(nullptr, 1024, "kernel");
                    checked_kernel_launch("kernel");
                    return 0;
                }}
                """
            ).lstrip(),
            encoding="utf-8",
        )
        cls.binary = root / "harness"
        subprocess.run(
            [compiler, "-std=c++17", "-I", str(root), str(harness), "-o", str(cls.binary)],
            check=True,
            capture_output=True,
            text=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_temporary"):
            cls._temporary.cleanup()

    def _run(self, mode: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(self.binary), mode], capture_output=True, text=True, check=False
        )

    def test_success_statuses_continue(self) -> None:
        self.assertEqual(self._run("success").returncode, 0)

    def test_injected_attribute_failure_raises(self) -> None:
        self.assertEqual(self._run("attribute_failure").returncode, 0)

    def test_injected_launch_failure_raises(self) -> None:
        self.assertEqual(self._run("launch_failure").returncode, 0)

    def test_header_checks_both_cuda_calls(self) -> None:
        source = HEADER.read_text(encoding="utf-8")
        self.assertIn("cudaFuncSetAttribute", source)
        self.assertIn("cudaGetLastError", source)
        self.assertGreaterEqual(source.count("TORCH_CHECK"), 2)


if __name__ == "__main__":
    unittest.main()
