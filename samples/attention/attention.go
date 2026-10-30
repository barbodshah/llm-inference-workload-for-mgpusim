package main

import (
	"flag"
	"github.com/sarchlab/mgpusim/v3/benchmarks/attention"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
	"log"
	"strconv"
	"strings"
)

func main() {
	c := attention.DefaultConfig()
	flag.IntVar(&c.Heads, "heads", c.Heads, "Query heads")
	flag.IntVar(&c.KVHeads, "kv-heads", c.KVHeads, "KV heads; must divide query heads")
	flag.IntVar(&c.Dim, "head-dim", c.Dim, "Head dimension, 1..64")
	flag.IntVar(&c.Chunk, "chunk", c.Chunk, "Number of prefill query tokens")
	flag.IntVar(&c.PrefillLength, "prefill-context", c.PrefillLength, "Prefill KV context including chunk")
	flag.StringVar(&c.Mode, "mode", c.Mode, "serial or concurrent")
	flag.StringVar(&c.Phase, "phase", c.Phase, "both, prefill, or decode; isolate phases for profiling")
	lengths := flag.String("decode-contexts", "17,9,13", "Comma-separated KV lengths, one per decode request")
	flag.Parse()
	c.DecodeLengths = nil
	for _, s := range strings.Split(*lengths, ",") {
		n, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil {
			log.Fatal(err)
		}
		c.DecodeLengths = append(c.DecodeLengths, n)
	}
	if err := c.Validate(); err != nil {
		log.Fatal(err)
	}
	gpus := flag.Lookup("gpus").Value.String()
	if (gpus != "" && gpus != "1") || flag.Lookup("unified-gpus").Value.String() != "" {
		log.Fatal("attention requires physical GPU 1")
	}
	if _, err := attention.LoadKernels(); err != nil {
		log.Fatal(err)
	}
	r := new(runner.Runner).ParseFlag().Init()
	b := attention.NewBenchmark(r.Driver())
	b.Config = c
	r.AddBenchmark(b)
	r.Run()
}
