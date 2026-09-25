// SPDX-License-Identifier: GPL-3.0-only
package main

import (
	"flag"
	"log"
	"strconv"
	"strings"

	"github.com/sarchlab/mgpusim/v3/benchmarks/llama"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

func main() {
	checkpoint := flag.String("checkpoint", "", "Legacy FP32 llama2.c checkpoint; empty uses a tiny synthetic model")
	kernelPath := flag.String("kernels", "", "Override the embedded gfx803 kernel binary")
	tokens := flag.String("tokens", "1,2,3", "Comma-separated token IDs consumed sequentially from position zero")
	phases := flag.String("phase-metrics", "", "Optional CSV path for initialization and per-token simulation/wall times")
	flag.Parse()
	model := llama.TinyModel()
	if *checkpoint != "" {
		var err error
		model, err = llama.LoadModel(*checkpoint)
		if err != nil {
			log.Fatal(err)
		}
	}
	var ids []int
	for _, s := range strings.Split(*tokens, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || n < 0 || n >= int(model.Config.VocabSize) {
			log.Fatalf("invalid token ID %q", s)
		}
		ids = append(ids, n)
	}
	if len(ids) > int(model.Config.SeqLen) {
		log.Fatal("token sequence exceeds checkpoint context length")
	}
	// Reject unified/multiple GPU selection explicitly, before runner setup.
	if flag.Lookup("unified-gpus").Value.String() != "" || strings.Contains(flag.Lookup("gpus").Value.String(), ",") {
		log.Fatal("Llama currently supports one physical simulated GPU only")
	}
	// Validate code objects before starting the simulator and its monitor.
	preflight := &llama.Benchmark{KernelPath: *kernelPath}
	if err := preflight.LoadKernels(); err != nil {
		log.Fatal(err)
	}
	r := new(runner.Runner).ParseFlag().Init()
	b := llama.NewBenchmark(r.Driver())
	b.Model = model
	b.Tokens = ids
	b.KernelPath = *kernelPath
	b.PhaseMetricsPath = *phases
	r.AddBenchmark(b)
	r.Run()
}
