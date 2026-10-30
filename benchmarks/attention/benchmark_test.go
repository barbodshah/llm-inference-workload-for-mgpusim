package attention

import (
	"github.com/sarchlab/akita/v3/tracing"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
	"math"
	"testing"
)

func TestKernelABI(t *testing.T) {
	kernels, err := LoadKernels()
	if err != nil {
		t.Fatal(err)
	}
	for name, k := range kernels {
		t.Logf("%s: kernarg=%d SGPR=%d VGPR=%d LDS=%d scratch=%d", name, k.KernargSegmentByteSize, k.WFSgprCount, k.WIVgprCount, k.WGGroupSegmentByteSize, k.WIPrivateSegmentByteSize)
	}
}

func TestAttentionGPU(t *testing.T) {
	cases := []Config{
		{Heads: 2, KVHeads: 2, Dim: 1, Chunk: 1, PrefillLength: 1, DecodeLengths: []int{1, 9}, Mode: "serial"},
		{Heads: 4, KVHeads: 2, Dim: 7, Chunk: 5, PrefillLength: 17, DecodeLengths: []int{17, 1, 9}, Mode: "concurrent"},
		{Heads: 4, KVHeads: 1, Dim: 32, Chunk: 8, PrefillLength: 33, DecodeLengths: []int{7, 33}, Mode: "serial"},
		{Heads: 2, KVHeads: 1, Dim: 64, Chunk: 3, PrefillLength: 257, DecodeLengths: []int{257, 65}, Mode: "concurrent"},
	}
	for i, c := range cases {
		t.Run(string(rune('A'+i)), func(t *testing.T) {
			p := runner.MakeEmuBuilder().WithNumGPU(1).Build()
			b := NewBenchmark(p.Driver)
			b.Config = c
			b.EnableVerification()
			if i == 2 {
				b.SetUnifiedMemory()
			}
			p.Driver.Run()
			defer p.Engine.Finished()
			defer p.Driver.Terminate()
			b.Run()
			b.Verify()
		})
	}
	for _, phase := range []string{"prefill", "decode"} {
		t.Run(phase, func(t *testing.T) {
			p := runner.MakeEmuBuilder().WithNumGPU(1).Build()
			b := NewBenchmark(p.Driver)
			b.Config.Phase = phase
			b.EnableVerification()
			p.Driver.Run()
			defer p.Engine.Finished()
			defer p.Driver.Terminate()
			b.Run()
			b.Verify()
			if len(b.data) != 1 {
				t.Fatal("phase filter did not isolate one kernel")
			}
		})
	}
}

func TestOnlineSoftmaxLargeScores(t *testing.T) {
	p := runner.MakeEmuBuilder().WithNumGPU(1).Build()
	b := NewBenchmark(p.Driver)
	programs, err := LoadKernels()
	if err != nil {
		t.Fatal(err)
	}
	p.Driver.Run()
	defer p.Engine.Finished()
	defer p.Driver.Terminate()
	b.Driver.SelectGPU(b.context, 1)
	d := makeDataset(5, 2, 1, 7, []int{17}, 51)
	for i := range d.q {
		d.q[i] *= 100
	}
	for i := range d.k {
		d.k[i] *= 100
	}
	b.initDataset(d)
	defer func() {
		for _, ptr := range d.allocations {
			_ = b.Driver.FreeMemory(b.context, ptr)
		}
	}()
	b.Driver.LaunchKernel(b.context, programs["attention_prefill"], [3]uint32{2 * 2 * 256, 1, 1}, [3]uint16{256, 1, 1}, &d.args)
	b.Driver.MemCopyD2H(b.context, d.out, d.args.Out)
	if err := compare(d.out, d.reference()); err != nil {
		t.Fatal(err)
	}
}

func TestDetailedAttention(t *testing.T) {
	p := runner.MakeR9NanoBuilder().WithNumGPU(1).Build()
	b := NewBenchmark(p.Driver)
	b.EnableVerification()
	b.Config.Mode = "concurrent"
	trace := NewResidencyTracer(p.Engine)
	for _, cu := range p.GPUs[0].CUs {
		tracing.CollectTrace(cu, trace)
	}
	p.Driver.Run()
	defer p.Engine.Finished()
	defer p.Driver.Terminate()
	b.Run()
	b.Verify()
	if len(trace.Pending) != 0 || len(trace.Completed) != 16 {
		t.Fatalf("residency pending=%d completed=%d", len(trace.Pending), len(trace.Completed))
	}
	t.Logf("attention workgroups=%d mixed CUs=%d (no overlap requirement for tiny attention)", len(trace.Completed), trace.MixedCUCount())
}

func TestConcurrencyAudit(t *testing.T) {
	for _, mode := range []string{"same-queue", "separate-queues"} {
		t.Run(mode, func(t *testing.T) {
			r, err := RunConcurrencyAudit(mode, 4, 512)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("launches=%d completed=%d mixed CUs=%d", r.Launches, r.Workgroups, r.MixedCUs)
		})
	}
}

func TestConfigRejectsInvalidInputs(t *testing.T) {
	c := DefaultConfig()
	mutations := []func(*Config){func(c *Config) { c.Heads = 3 }, func(c *Config) { c.KVHeads = 0 }, func(c *Config) { c.Dim = 65 }, func(c *Config) { c.Chunk = 17 }, func(c *Config) { c.DecodeLengths = []int{0} }, func(c *Config) { c.Mode = "adaptive" }, func(c *Config) { c.Heads = 4096; c.KVHeads = 4096; c.Dim = 64; c.PrefillLength = 1 << 20 }}
	for i, mutate := range mutations {
		x := c
		mutate(&x)
		if x.Validate() == nil {
			t.Errorf("case %d accepted", i)
		}
	}
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
	if compare([]float32{float32(math.NaN())}, []float32{0}) == nil {
		t.Fatal("accepted NaN")
	}
}
