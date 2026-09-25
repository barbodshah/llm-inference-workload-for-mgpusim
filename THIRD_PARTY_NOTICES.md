# Third-party notices

This repository combines code under multiple licenses. The location of a file
determines the applicable license and notice.

## MGPUSim

The simulator source derives from MGPUSim 3.0.2 by the Project Akita authors.
It is distributed under the MIT license reproduced in the root `LICENSE` file.
The original project overview is preserved in `docs/MGPUSIM_UPSTREAM_README.md`.

## Llama inference benchmark

Files under `benchmarks/llama/` and `samples/llama/` adapt the FP32 inference
algorithm and HIP kernels from:

- https://github.com/Uzair-90/llama-inference-hip
- https://github.com/karpathy/llama2.c

They are marked GPL-3.0-only. The GPL text and detailed attribution are in
`benchmarks/llama/LICENSE` and `benchmarks/llama/NOTICE`.

## ROCm device libraries

The committed `benchmarks/llama/kernels.hsaco` links code from AMD ROCm device
libraries 5.0.0. Their copyright and redistribution terms are reproduced in
`benchmarks/llama/ROCM-DEVICE-LIBS-LICENSE`.

No model checkpoint or tokenizer data is distributed in this repository.

