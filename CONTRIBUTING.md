# Contributing

Thank you for improving the LLM inference workload for MGPUSim.

## Development setup

Install Go 1.22 or newer, clone the repository, and run:

```bash
go test ./benchmarks/llama ./emu ./insts ./driver ./kernels ./timing/cu
go vet ./benchmarks/llama ./samples/llama
```

Keep model checkpoints, generated metrics, executables, and traces out of Git.
Use the deterministic synthetic model for routine development and add a focused
test for changes to checkpoint parsing, tensor indexing, kernel arguments,
numerical behavior, or simulator instructions.

## Kernel changes

Kernel changes require Ubuntu/Linux with the compiler versions documented in
`benchmarks/llama/README.md`. Rebuild `kernels.hsaco`, inspect the generated
disassembly, rerun all benchmark and emulator tests, and update the hashes and
evidence in `benchmarks/llama/VALIDATION.md`.

Do not commit generated disassembly. The HSACO is committed because it is the
embedded runnable workload and lets users run without ROCm.

## Pull requests

Explain the behavior change, its reason, and the validation performed. Keep
unrelated formatting or refactoring out of focused changes. Preserve the
licenses and notices for MGPUSim, llama-inference-hip, llama2.c, and ROCm device
libraries.

