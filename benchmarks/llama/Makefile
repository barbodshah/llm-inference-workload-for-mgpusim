# Validated with Ubuntu 22.04 Clang/LLD 14 and rocm-device-libs 5.0.0.

.PHONY: all disasm check
all: kernels.hsaco

kernels.hsaco: kernels.cl build-kernels.sh
	bash build-kernels.sh

disasm:
	bash build-kernels.sh

check: kernels.hsaco
	cd ../../samples/llama && go run . -verify -tokens 1,2,3
	cd ../../samples/llama && go run . -verify -timing -tokens 1,2,3
