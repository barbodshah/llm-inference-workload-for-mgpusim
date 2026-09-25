# Validation record — 2026-09-26

## Reproducibility

- Windows amd64, Go 1.27.1; WSL2 Ubuntu 22.04.5.
- Clang/LLD 14.0.0-1ubuntu1.1; rocm-device-libs 5.0.0-1.
- gfx803, code object v2, wave64; flags recorded in build-kernels.sh.
- kernels.cl SHA256:
  `4902825d09cc63ef7a1e24835ed3174415e9f3f533afe856d787ac6d6615cd97`
- kernels.hsaco SHA256:
  `858eba2ad7ebe53d310a97cd1834168901ca8745415166e6ff627edab2e64210`
- stories15M.bin: 60,816,028 bytes, SHA256:
  `cd590644d963867a2b6e5a1107f51fad663c41d79c149fbecbbb1f95fa81f49a`
- Original run.c SHA256:
  `9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd`

Checkpoint source:
https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
The downloaded checkpoint remains external in the workspace .build-cache.

## Passed

1. Checkpoint/CPU tests: both classifier layouts, RoPE-table skipping, malformed
   input rejection, overflow checks, analytic residual case, persistent KV history.
2. GPU kernels at sizes 2, 65, 257, 513: matmul, RMSNorm (also in-place), SwiGLU,
   and residual addition. GQA attention at positions 0, 1, 129, 256.
3. Full two-layer synthetic inference, tokens [1,2,3], with logits/KV checks,
   repeated for normal and unified allocations.
4. Synthetic detailed timing simulation with verification.
5. Real stories15M functional simulation of [1,2,3], checking logits and KV.
   Argmax outputs: [9038,471,471]. Inputs are fixed IDs, not sampled text.
6. Go reference versus original C implementation for synthetic shared/unshared
   classifiers and all three positions of stories15M.
7. Emulator and decoder regression suites, Go vet, and Llama executable build.
8. Synthetic performance run with timing/report-all and no verification,
   producing runner metrics and per-token phase timings.

## Simulator compatibility changes

The compiler exposed missing instructions in this MGPUSim snapshot:

- V_CMP_EQ_F32 and V_CMP_GE_F32 in VOP3 encoding: ordered comparisons with
  execution masks and the existing source-modifier handling.
- S_CSELECT_B64: dispatch support and corrected 64-bit decoder operand widths.
- S_BFE_U32: unsigned bit extraction with SCC updates.

New tests cover NaN, infinity, signed zero, masked lanes, absolute source
modifiers, both SCC select cases, high-half payloads, decoded register pairs,
and extraction boundary widths. Existing timing classes are retained; no new
hardware latency model is introduced. Instruction reference:
https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/gcn3-instruction-set-architecture.pdf

## Limits

This validates functionality and integration, not physical-GPU performance
calibration. Full stories15M detailed timing, larger models, long-context full
inference, text generation, and multi-GPU partitioning are not covered.
Changing compiler versions requires renewed ABI, ISA, and numerical checks.
