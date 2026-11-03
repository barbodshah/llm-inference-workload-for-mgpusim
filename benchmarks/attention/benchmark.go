// SPDX-License-Identifier: MIT
package attention

import (
	"bytes"
	"debug/elf"
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/insts"
)

//go:embed kernels.hsaco
var kernelBinary []byte

// Config describes independent requests. Length includes the current query
// tokens. Prefill queries occupy the last Chunk positions of its KV cache.
type Config struct {
	CompactDecode                             bool
	Heads, KVHeads, Dim, Chunk, PrefillLength int
	DecodeLengths                             []int
	Mode                                      string
	Phase                                     string
}

func DefaultConfig() Config {
	return Config{Heads: 4, KVHeads: 2, Dim: 8, Chunk: 4, PrefillLength: 16,
		DecodeLengths: []int{17, 9, 13}, Mode: "serial", Phase: "both"}
}

func (c Config) Validate() error {
	if c.CompactDecode && c.Dim != 8 {
		return fmt.Errorf("compact decode requires head dimension 8")
	}
	if c.Heads < 1 || c.Heads > 4096 || c.KVHeads < 1 || c.Heads%c.KVHeads != 0 {
		return fmt.Errorf("heads must be 1..4096 and divisible by positive KV heads")
	}
	if c.Dim < 1 || c.Dim > 64 {
		return fmt.Errorf("head dimension must be 1..64")
	}
	if c.Chunk < 1 || c.PrefillLength < c.Chunk || c.PrefillLength > 1<<20 {
		return fmt.Errorf("require 1 <= chunk <= prefill length <= 1048576")
	}
	if len(c.DecodeLengths) == 0 || len(c.DecodeLengths) > 65536 {
		return fmt.Errorf("require 1..65536 decode requests")
	}
	maxLen := 0
	for _, n := range c.DecodeLengths {
		if n < 1 || n > 1<<20 {
			return fmt.Errorf("decode lengths must be 1..1048576")
		}
		if n > maxLen {
			maxLen = n
		}
	}
	// Bound element counts and all kernel-side signed 32-bit address arithmetic.
	for _, n := range []int64{int64(c.Chunk) * int64(c.Heads) * int64(c.Dim),
		int64(c.PrefillLength) * int64(c.KVHeads) * int64(c.Dim),
		int64(len(c.DecodeLengths)) * int64(maxLen) * int64(c.KVHeads) * int64(c.Dim),
		int64(len(c.DecodeLengths)) * int64(c.Heads) * int64(c.Dim)} {
		if n > 1<<28 {
			return fmt.Errorf("tensor exceeds research benchmark limit of 2^28 FP32 elements")
		}
	}
	if uint64(c.Heads)*uint64((c.Chunk+3)/4)*256 > math.MaxUint32 || uint64(len(c.DecodeLengths))*uint64(c.Heads)*64 > math.MaxUint32 {
		return fmt.Errorf("launch grid exceeds 32-bit dispatch dimensions")
	}
	if c.Mode != "serial" && c.Mode != "concurrent" {
		return fmt.Errorf("mode must be serial or concurrent")
	}
	if c.Phase != "" && c.Phase != "both" && c.Phase != "prefill" && c.Phase != "decode" {
		return fmt.Errorf("phase must be both, prefill, or decode")
	}
	return nil
}

type attentionArgs struct {
	Out, Q, K, V, Lengths                driver.Ptr
	Queries, Heads, KVHeads, Dim, Stride int32
}
type probeArgs struct {
	Out          driver.Ptr
	Rounds, Seed uint32
}

// LoadKernels validates the actual compiler ABI before any execution.
func LoadKernels() (map[string]*insts.HsaCo, error) {
	f, err := elf.NewFile(bytes.NewReader(kernelBinary))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sec := f.Section(".text")
	if sec == nil {
		return nil, fmt.Errorf("missing .text")
	}
	data, err := sec.Data()
	if err != nil {
		return nil, err
	}
	symbols, err := f.Symbols()
	if err != nil {
		return nil, err
	}
	out := make(map[string]*insts.HsaCo)
	for name, arg := range map[string]interface{}{"attention_prefill": attentionArgs{}, "attention_decode": attentionArgs{}, "attention_decode_compact": attentionArgs{}, "concurrency_probe": probeArgs{}} {
		for _, s := range symbols {
			if s.Name != name {
				continue
			}
			if s.Value < sec.Addr || s.Value-sec.Addr > uint64(len(data)) || s.Size < 256 || s.Size > uint64(len(data))-(s.Value-sec.Addr) {
				return nil, fmt.Errorf("invalid extent %s", name)
			}
			start := s.Value - sec.Addr
			co := insts.NewHsaCoFromData(data[start : start+s.Size])
			if co.MachineVersionMajor != 8 || co.MachineVersionMinor != 0 || co.MachineVersionStepping != 3 || co.KernelCodeEntryByteOffset != 256 || co.WavefrontSize != 6 || co.WIPrivateSegmentByteSize != 0 {
				return nil, fmt.Errorf("%s requires scratch-free legacy gfx803 wave64 code", name)
			}
			if co.KernargSegmentByteSize != uint64(binary.Size(arg)) {
				return nil, fmt.Errorf("%s kernarg size %d, host %d", name, co.KernargSegmentByteSize, binary.Size(arg))
			}
			out[name] = co
		}
		if out[name] == nil {
			return nil, fmt.Errorf("missing kernel %s", name)
		}
	}
	return out, nil
}

type dataset struct {
	queries, heads, kvHeads, dim, stride int
	lengths                              []int32
	q, k, v                              []float32
	out                                  []float32
	args                                 attentionArgs
	allocations                          []driver.Ptr
}

func makeDataset(queries, heads, kvHeads, dim int, lengths []int, seed uint32) *dataset {
	d := &dataset{queries: queries, heads: heads, kvHeads: kvHeads, dim: dim}
	for _, n := range lengths {
		d.lengths = append(d.lengths, int32(n))
		if n > d.stride {
			d.stride = n
		}
	}
	d.q = make([]float32, len(lengths)*queries*heads*dim)
	d.k = make([]float32, len(lengths)*d.stride*kvHeads*dim)
	d.v = make([]float32, len(d.k))
	d.out = make([]float32, len(d.q))
	next := func() float32 { seed = seed*1664525 + 1013904223; return float32(int(seed>>16)-32768) / 32768 }
	for i := range d.q {
		d.q[i] = next()
	}
	for i := range d.k {
		d.k[i] = next()
		d.v[i] = next()
	}
	// Poison ragged padding: incorrect context handling must not silently pass.
	for r, n := range lengths {
		for t := n; t < d.stride; t++ {
			for x := 0; x < kvHeads*dim; x++ {
				i := (r*d.stride+t)*kvHeads*dim + x
				d.k[i] = 10000
				d.v[i] = -10000
			}
		}
	}
	return d
}

func (d *dataset) reference() []float32 {
	out := make([]float32, len(d.q))
	for r, n := range d.lengths {
		for query := 0; query < d.queries; query++ {
			for h := 0; h < d.heads; h++ {
				limit := int(n) - d.queries + query + 1
				scores := make([]float64, limit)
				maximum := math.Inf(-1)
				qbase := ((r*d.queries+query)*d.heads + h) * d.dim
				kh := h / (d.heads / d.kvHeads)
				for t := 0; t < limit; t++ {
					kbase := ((r*d.stride+t)*d.kvHeads + kh) * d.dim
					var dot float64
					for x := 0; x < d.dim; x++ {
						dot += float64(d.q[qbase+x]) * float64(d.k[kbase+x])
					}
					scores[t] = dot / math.Sqrt(float64(d.dim))
					maximum = math.Max(maximum, scores[t])
				}
				var sum float64
				for t := range scores {
					scores[t] = math.Exp(scores[t] - maximum)
					sum += scores[t]
				}
				for x := 0; x < d.dim; x++ {
					var acc float64
					for t, w := range scores {
						acc += w * float64(d.v[((r*d.stride+t)*d.kvHeads+kh)*d.dim+x])
					}
					out[qbase+x] = float32(acc / sum)
				}
			}
		}
	}
	return out
}

func compare(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("output length mismatch")
	}
	for i, g := range got {
		w := want[i]
		if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) || math.Abs(float64(g-w)) > 1e-4+1e-3*math.Abs(float64(w)) {
			return fmt.Errorf("element %d: got %g, want %g", i, g, w)
		}
	}
	return nil
}

type Benchmark struct {
	Driver          *driver.Driver
	Config          Config
	context         *driver.Context
	unified, verify bool
	data            []*dataset
}

func NewBenchmark(d *driver.Driver) *Benchmark {
	return &Benchmark{Driver: d, Config: DefaultConfig(), context: d.Init()}
}
func (b *Benchmark) SelectGPU(ids []int) {
	if len(ids) != 1 || ids[0] != 1 {
		panic("attention requires physical GPU 1")
	}
}
func (b *Benchmark) SetUnifiedMemory()   { b.unified = true }
func (b *Benchmark) EnableVerification() { b.verify = true }

func (b *Benchmark) initDataset(d *dataset) {
	alloc := func(n int) driver.Ptr {
		var p driver.Ptr
		if b.unified {
			p = b.Driver.AllocateUnifiedMemory(b.context, uint64(n)*4)
		} else {
			p = b.Driver.AllocateMemory(b.context, uint64(n)*4)
		}
		d.allocations = append(d.allocations, p)
		return p
	}
	d.args = attentionArgs{Out: alloc(len(d.out)), Q: alloc(len(d.q)), K: alloc(len(d.k)), V: alloc(len(d.v)), Lengths: alloc(len(d.lengths)), Queries: int32(d.queries), Heads: int32(d.heads), KVHeads: int32(d.kvHeads), Dim: int32(d.dim), Stride: int32(d.stride)}
	b.Driver.MemCopyH2D(b.context, d.args.Q, d.q)
	b.Driver.MemCopyH2D(b.context, d.args.K, d.k)
	b.Driver.MemCopyH2D(b.context, d.args.V, d.v)
	b.Driver.MemCopyH2D(b.context, d.args.Lengths, d.lengths)
	poison := make([]float32, len(d.out))
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	b.Driver.MemCopyH2D(b.context, d.args.Out, poison)
}

func (b *Benchmark) Run() {
	if err := b.Config.Validate(); err != nil {
		panic(err)
	}
	programs, err := LoadKernels()
	if err != nil {
		panic(err)
	}
	b.Driver.SelectGPU(b.context, 1)
	c := b.Config
	prefill := makeDataset(c.Chunk, c.Heads, c.KVHeads, c.Dim, []int{c.PrefillLength}, 1)
	decode := makeDataset(1, c.Heads, c.KVHeads, c.Dim, c.DecodeLengths, 7)
	b.data = []*dataset{prefill, decode}
	if c.Phase == "prefill" {
		b.data = []*dataset{prefill}
	}
	if c.Phase == "decode" {
		b.data = []*dataset{decode}
	}
	for _, d := range b.data {
		b.initDataset(d)
	}
	defer func() {
		for _, d := range b.data {
			for _, p := range d.allocations {
				if err := b.Driver.FreeMemory(b.context, p); err != nil {
					panic(err)
				}
			}
		}
	}()
	queues := make([]*driver.CommandQueue, len(b.data))
	for i := range queues {
		queues[i] = b.Driver.CreateCommandQueue(b.context)
	}
	launches := make([]driver.Command, len(b.data))
	for i, d := range b.data {
		qt, name := 4, "attention_prefill"
		if d == decode {
			qt, name = 1, "attention_decode"
			if c.CompactDecode {
				name = "attention_decode_compact"
			}
		}
		groups := len(d.lengths) * d.heads * ((d.queries + qt - 1) / qt)
		launches[i] = prepareLaunch(b.Driver, b.context, programs[name], [3]uint32{uint32(groups * qt * 64), 1, 1}, [3]uint16{uint16(qt * 64), 1, 1}, &d.args)
	}
	for i, cmd := range launches {
		queues[i].Enqueue(cmd)
		if c.Mode == "serial" {
			b.Driver.DrainCommandQueue(queues[i])
		}
	}
	if c.Mode == "concurrent" {
		for _, q := range queues {
			b.Driver.DrainCommandQueue(q)
		}
	}
	if b.verify {
		for _, d := range b.data {
			b.Driver.MemCopyD2H(b.context, d.out, d.args.Out)
		}
	}
	fmt.Fprintf(os.Stdout, "attention: phase=%s mode=%s heads=%d kv_heads=%d dim=%d chunk=%d prefill_context=%d decode_contexts=%v verification=%t\n", c.Phase, c.Mode, c.Heads, c.KVHeads, c.Dim, c.Chunk, c.PrefillLength, c.DecodeLengths, b.verify)
}

func (b *Benchmark) Verify() {
	if !b.verify {
		panic("verification must be enabled before Run")
	}
	for i, d := range b.data {
		if err := compare(d.out, d.reference()); err != nil {
			panic(fmt.Errorf("attention dataset %d: %w", i, err))
		}
	}
	fmt.Println("attention: CPU reference verification passed")
}
