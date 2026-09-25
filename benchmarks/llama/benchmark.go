// SPDX-License-Identifier: GPL-3.0-only
package llama

import (
	"bytes"
	"debug/elf"
	_ "embed"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/insts"
)

//go:embed kernels.hsaco
var embeddedKernels []byte

// Benchmark runs a fixed token sequence on one simulated GPU.
type Benchmark struct {
	driver          *driver.Driver
	context         *driver.Context
	gpus            []int
	unified, verify bool
	Model           *Model
	Tokens          []int
	KernelPath      string
	Logits          [][]float32
	programs        map[string]*insts.HsaCo
	weights         map[string]driver.Ptr
	state           map[string]driver.Ptr
	// PhaseMetricsPath optionally receives initialization and per-token timings.
	PhaseMetricsPath string
	Phases           []PhaseMetric
}

func NewBenchmark(d *driver.Driver) *Benchmark {
	return &Benchmark{driver: d, context: d.Init(), gpus: []int{1}, Model: TinyModel(), Tokens: []int{1, 2, 3}}
}
func (b *Benchmark) SelectGPU(ids []int) {
	if len(ids) != 1 {
		panic("llama currently requires exactly one GPU")
	}
	b.gpus = append([]int(nil), ids...)
}
func (b *Benchmark) SetUnifiedMemory()   { b.unified = true }
func (b *Benchmark) EnableVerification() { b.verify = true }

// Argument layouts match Clang 14 / ROCm device libs 5.0 code-object v2.
// Kernels using global IDs retain seven implicit 64-bit arguments. The
// compiler removes implicit arguments entirely from RMSNorm and attention.
type matArgs struct {
	Out, X, W                 driver.Ptr
	N, D                      int32
	OffsetX, OffsetY, OffsetZ int64
	Reserved                  [4]uint64
}
type normArgs struct {
	Out, X, W driver.Ptr
	N         int32
}
type pairArgs struct {
	X, Y                      driver.Ptr
	N                         int32
	Padding                   uint32
	OffsetX, OffsetY, OffsetZ int64
	Reserved                  [4]uint64
}
type ropeArgs struct {
	Q, K                      driver.Ptr
	Dim, KV, Head, Pos        int32
	OffsetX, OffsetY, OffsetZ int64
	Reserved                  [4]uint64
}
type attentionArgs struct {
	Out, Q, Keys, Values, Att            driver.Ptr
	Head, KV, Mul, Seq, Pos, LayerOffset int32
}

// LoadKernels checks the legacy kernel header before the simulator sees it.
// Modern descriptor-based code objects cannot be used with this MGPUSim loader.
func (b *Benchmark) LoadKernels() error {
	data := embeddedKernels
	if b.KernelPath != "" {
		var err error
		data, err = os.ReadFile(b.KernelPath)
		if err != nil {
			return fmt.Errorf("read kernels: %w", err)
		}
	}
	f, err := elf.NewFile(bytes.NewReader(data))
	if err != nil {
		return err
	}
	defer f.Close()
	section := f.Section(".text")
	if section == nil {
		return fmt.Errorf("HSACO has no .text section")
	}
	code, err := section.Data()
	if err != nil {
		return err
	}
	symbols, err := f.Symbols()
	if err != nil {
		return err
	}
	args := map[string]interface{}{"kernel_matmul": matArgs{}, "kernel_rmsnorm": normArgs{}, "kernel_add": pairArgs{}, "kernel_swiglu": pairArgs{}, "kernel_rope": ropeArgs{}, "kernel_attention": attentionArgs{}}
	programs := make(map[string]*insts.HsaCo)
	for name, arg := range args {
		for _, symbol := range symbols {
			if symbol.Name != name {
				continue
			}
			if symbol.Value < section.Addr {
				return fmt.Errorf("invalid symbol %s", name)
			}
			off := symbol.Value - section.Addr
			if off > uint64(len(code)) || symbol.Size < 256 || symbol.Size > uint64(len(code))-off {
				return fmt.Errorf("invalid kernel extent for %s", name)
			}
			co := insts.NewHsaCoFromData(code[off : off+symbol.Size])
			if co.MachineVersionMajor != 8 || co.MachineVersionMinor != 0 || co.MachineVersionStepping != 3 || co.KernelCodeEntryByteOffset != 256 || co.WavefrontSize != 6 {
				return fmt.Errorf("%s: expected legacy gfx803 header, 256-byte entry offset, wave64", name)
			}
			if co.WIPrivateSegmentByteSize != 0 {
				return fmt.Errorf("%s: scratch/private memory unsupported by this benchmark", name)
			}
			if co.KernargSegmentByteSize != uint64(binary.Size(arg)) {
				return fmt.Errorf("%s: kernarg size %d, expected %d; inspect compiler ABI", name, co.KernargSegmentByteSize, binary.Size(arg))
			}
			programs[name] = co
			break
		}
		if programs[name] == nil {
			return fmt.Errorf("kernel %s missing (need legacy OpenCL symbols, not modern .kd descriptors)", name)
		}
	}
	b.programs = programs
	return nil
}

func (b *Benchmark) alloc(n int) driver.Ptr {
	if b.unified {
		return b.driver.AllocateUnifiedMemory(b.context, uint64(n)*4)
	}
	return b.driver.AllocateMemory(b.context, uint64(n)*4)
}
func ptr(base driver.Ptr, elements int) driver.Ptr { return base + driver.Ptr(uint64(elements)*4) }
func (b *Benchmark) initMem() {
	m := b.Model
	c := m.Config
	b.weights = make(map[string]driver.Ptr)
	b.state = make(map[string]driver.Ptr)
	for _, t := range []struct {
		name string
		data []float32
	}{{"emb", m.Embedding}, {"an", m.AttNorm}, {"q", m.Q}, {"k", m.K}, {"v", m.V}, {"o", m.O}, {"fn", m.FFNNorm}, {"w1", m.W1}, {"w2", m.W2}, {"w3", m.W3}, {"final", m.FinalNorm}} {
		p := b.alloc(len(t.data))
		b.weights[t.name] = p
		b.driver.MemCopyH2D(b.context, p, t.data)
	}
	if &m.Classifier[0] == &m.Embedding[0] {
		b.weights["cls"] = b.weights["emb"]
	} else {
		p := b.alloc(len(m.Classifier))
		b.weights["cls"] = p
		b.driver.MemCopyH2D(b.context, p, m.Classifier)
	}
	d, h := int(c.Dim), int(c.HiddenDim)
	cache := int(c.Layers) * int(c.SeqLen) * int(c.kvDim())
	for _, t := range []struct {
		name string
		n    int
	}{{"x", d}, {"xb", d}, {"xb2", d}, {"q", d}, {"hb", h}, {"hb2", h}, {"att", int(c.Heads) * int(c.SeqLen)}, {"logits", int(c.VocabSize)}, {"keys", cache}, {"values", cache}} {
		p := b.alloc(t.n)
		b.state[t.name] = p
		if t.name == "keys" || t.name == "values" {
			b.driver.MemCopyH2D(b.context, p, make([]float32, t.n))
		}
	}
}
func (b *Benchmark) launch(name string, n, group int, args interface{}) {
	global := uint32((n + group - 1) / group * group)
	b.driver.LaunchKernel(b.context, b.programs[name], [3]uint32{global, 1, 1}, [3]uint16{uint16(group), 1, 1}, args)
}
func (b *Benchmark) mat(out, x, w driver.Ptr, n, d int) {
	b.launch("kernel_matmul", d, 256, &matArgs{Out: out, X: x, W: w, N: int32(n), D: int32(d)})
}
func (b *Benchmark) norm(out, x, w driver.Ptr, n int) {
	b.launch("kernel_rmsnorm", 256, 256, &normArgs{Out: out, X: x, W: w, N: int32(n)})
}
func (b *Benchmark) pair(name string, x, y driver.Ptr, n int) {
	b.launch(name, n, 256, &pairArgs{X: x, Y: y, N: int32(n)})
}

func (b *Benchmark) forward(token, pos int) []float32 {
	c := b.Model.Config
	d, h, k, hs := int(c.Dim), int(c.HiddenDim), int(c.kvDim()), int(c.Dim/c.Heads)
	s, w := b.state, b.weights
	b.driver.MemCopyD2D(b.context, s["x"], ptr(w["emb"], token*d), d*4)
	for l := 0; l < int(c.Layers); l++ {
		off := l * int(c.SeqLen) * k
		kp, vp := ptr(s["keys"], off+pos*k), ptr(s["values"], off+pos*k)
		b.norm(s["xb"], s["x"], ptr(w["an"], l*d), d)
		b.mat(s["q"], s["xb"], ptr(w["q"], l*d*d), d, d)
		b.mat(kp, s["xb"], ptr(w["k"], l*d*k), d, k)
		b.mat(vp, s["xb"], ptr(w["v"], l*d*k), d, k)
		b.launch("kernel_rope", d/2, 128, &ropeArgs{Q: s["q"], K: kp, Dim: int32(d), KV: int32(k), Head: int32(hs), Pos: int32(pos)})
		b.launch("kernel_attention", int(c.Heads)*128, 128, &attentionArgs{Out: s["xb"], Q: s["q"], Keys: s["keys"], Values: s["values"], Att: s["att"], Head: int32(hs), KV: int32(k), Mul: c.Heads / c.KVHeads, Seq: c.SeqLen, Pos: int32(pos), LayerOffset: int32(off)})
		b.mat(s["xb2"], s["xb"], ptr(w["o"], l*d*d), d, d)
		b.pair("kernel_add", s["x"], s["xb2"], d)
		b.norm(s["xb"], s["x"], ptr(w["fn"], l*d), d)
		b.mat(s["hb"], s["xb"], ptr(w["w1"], l*d*h), d, h)
		b.mat(s["hb2"], s["xb"], ptr(w["w3"], l*d*h), d, h)
		b.pair("kernel_swiglu", s["hb"], s["hb2"], h)
		b.mat(s["xb"], s["hb"], ptr(w["w2"], l*d*h), h, d)
		b.pair("kernel_add", s["x"], s["xb"], d)
	}
	b.norm(s["x"], s["x"], w["final"], d)
	b.mat(s["logits"], s["x"], w["cls"], d, int(c.VocabSize))
	logits := make([]float32, int(c.VocabSize))
	b.driver.MemCopyD2H(b.context, logits, s["logits"])
	return logits
}

func (b *Benchmark) Run() {
	if err := b.Model.validate(); err != nil {
		panic(err)
	}
	if len(b.Tokens) == 0 || len(b.Tokens) > int(b.Model.Config.SeqLen) {
		panic("llama: token sequence must be nonempty and fit context")
	}
	for _, t := range b.Tokens {
		if t < 0 || t >= int(b.Model.Config.VocabSize) {
			panic("llama: token out of vocabulary")
		}
	}
	if b.programs == nil {
		if err := b.LoadKernels(); err != nil {
			panic(err)
		}
	}
	b.driver.SelectGPU(b.context, b.gpus[0])
	b.Phases = nil
	simStart, wallStart := b.phaseStart()
	b.initMem()
	b.phaseEnd("initialization", -1, simStart, wallStart)
	defer b.releaseMem()
	b.Logits = nil
	var ref *Reference
	if b.verify {
		ref = NewReference(b.Model)
	}
	for pos, t := range b.Tokens {
		simStart, wallStart = b.phaseStart()
		out := b.forward(t, pos)
		b.phaseEnd("forward", pos, simStart, wallStart)
		b.Logits = append(b.Logits, out)
		if ref != nil {
			simStart, wallStart = b.phaseStart()
			checkClose(fmt.Sprintf("position %d logits", pos), out, ref.Forward(t))
			for _, entry := range []struct {
				name string
				want []float32
			}{{"keys", ref.keys}, {"values", ref.values}} {
				got := make([]float32, len(entry.want))
				b.driver.MemCopyD2H(b.context, got, b.state[entry.name])
				checkClose(entry.name, got, entry.want)
			}
			b.phaseEnd("verification", pos, simStart, wallStart)
		}
		best := 0
		for i := range out {
			if out[i] > out[best] {
				best = i
			}
		}
		log.Printf("Llama position=%d input=%d argmax=%d", pos, t, best)
	}
	if b.PhaseMetricsPath != "" {
		if err := b.writePhases(); err != nil {
			panic(err)
		}
	}
}

func (b *Benchmark) phaseStart() (float64, time.Time) {
	return float64(b.driver.Engine.CurrentTime()), time.Now()
}
func (b *Benchmark) phaseEnd(name string, pos int, start float64, wall time.Time) {
	b.Phases = append(b.Phases, PhaseMetric{name, pos, float64(b.driver.Engine.CurrentTime()) - start, time.Since(wall).Seconds()})
}

func (b *Benchmark) releaseMem() {
	seen := make(map[driver.Ptr]bool)
	for _, allocations := range []map[string]driver.Ptr{b.weights, b.state} {
		for _, p := range allocations {
			if !seen[p] {
				if err := b.driver.FreeMemory(b.context, p); err != nil {
					panic(err)
				}
				seen[p] = true
			}
		}
	}
}
func checkClose(label string, got, want []float32) {
	if len(got) != len(want) {
		panic(label + ": length mismatch")
	}
	for i, x := range got {
		y := want[i]
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.IsNaN(float64(y)) || math.IsInf(float64(y), 0) || math.Abs(float64(x-y)) > 1e-4+1e-3*math.Abs(float64(y)) {
			panic(fmt.Sprintf("%s[%d]: got %g, expected %g", label, i, x, y))
		}
	}
}
func (b *Benchmark) Verify() {
	if len(b.Logits) != len(b.Tokens) {
		panic("llama: incomplete run")
	}
	ref := NewReference(b.Model)
	for i, t := range b.Tokens {
		checkClose(fmt.Sprintf("position %d", i), b.Logits[i], ref.Forward(t))
	}
	log.Print("Llama verification passed")
}
