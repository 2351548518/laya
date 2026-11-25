# laya-cpp

A C++17 port of the classic Laya text engine. The Python implementation is the reference for
tokenization, language detection, sequence construction and decision decoding. The C++ tests check these
results against recorded Python outputs. [Laya text routing and integrations](../docs/laya-guide.md)

The `laya` library accepts classic text questions. `LAYA_ONNX=ON` adds `laya_onnx` to run
exported Laya ONNX checkpoints. A C++ application that needs native Gemma or Qwen media decisions
can send media to the local Python HTTP service or the matching GGUF server.
[Native HTTP request](../docs/multimodal-clients.md#http-without-an-sdk) · [GGUF runtime](../docs/multimodal-release.md)

## Build and test

Needs CMake 3.25+ for the presets and a C++17 standard library with floating-point
`std::to_chars`. Configuration checks for the required overload. CI exercises GCC 13,
Clang 18 and Apple Clang. The first test configure fetches GoogleTest, pinned by commit.

```sh
cd laya-cpp
cmake --preset debug && cmake --build --preset debug && ctest --preset debug
# also: asan-ubsan, tsan, release
```

Options: `LAYA_BUILD_TESTS` (default ON), `LAYA_ONNX` (default OFF).

For an ONNX build, configure with `-DLAYA_ONNX=ON`. CMake downloads the pinned ONNX Runtime
archive or uses the local release selected by `-DLAYA_ONNXRUNTIME_ROOT=/path/to/onnxruntime`.

Floating point is compiled with `-ffp-contract=off -fno-fast-math` (`/fp:precise` on MSVC) and
warnings are errors. The `compile_flags` test fails if either flag is missing from a laya-cpp
translation unit.

## Test-count gate

`scripts/check_test_xml.py` reads JUnit XML and fails on any skipped or failed test and below
`--min-tests`; `scripts/test_check_test_xml.py` proves it can fail:

```sh
ctest --preset debug --output-junit "$PWD/build/debug/junit.xml"
python3 scripts/check_test_xml.py build/debug/junit.xml --min-tests 2
python3 scripts/test_check_test_xml.py
```
