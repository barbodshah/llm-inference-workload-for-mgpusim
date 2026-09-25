// SPDX-License-Identifier: GPL-3.0-only
package llama

import (
	"math"
	"os"
	"testing"

	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

// Executes the embedded code object. No CPU fallback can pass this test.
func TestGPUForward(t *testing.T) {
	path := os.Getenv("MGPUSIM_LLAMA_HSACO")
	platform := runner.MakeEmuBuilder().WithNumGPU(1).Build()
	b := NewBenchmark(platform.Driver)
	b.KernelPath = path
	b.EnableVerification()
	if err := b.LoadKernels(); err != nil {
		t.Fatal(err)
	}
	platform.Driver.Run()
	defer platform.Engine.Finished()
	defer platform.Driver.Terminate()
	b.Run()
	b.Verify()
	// Repeat on unified allocations, resetting KV state between runs.
	b.SetUnifiedMemory()
	b.Run()
	b.Verify()
}

func TestGPUKernels(t *testing.T) {
	path := os.Getenv("MGPUSIM_LLAMA_HSACO")
	platform := runner.MakeEmuBuilder().WithNumGPU(1).Build()
	b := NewBenchmark(platform.Driver)
	b.KernelPath = path
	if err := b.LoadKernels(); err != nil {
		t.Fatal(err)
	}
	platform.Driver.Run()
	defer platform.Engine.Finished()
	defer platform.Driver.Terminate()
	b.driver.SelectGPU(b.context, 1)
	// Cross-wave reductions, strided reduction reads, and partial final groups.
	for _, n := range []int{2, 65, 257, 513} {
		x, w := make([]float32, n), make([]float32, n)
		for i := range x {
			x[i] = float32(i%17-8) * 0.1
			w[i] = float32(i%5+1) * 0.2
		}
		xp, wp, out := b.alloc(n), b.alloc(n), b.alloc(n)
		b.driver.MemCopyH2D(b.context, xp, x)
		b.driver.MemCopyH2D(b.context, wp, w)
		got := make([]float32, n)
		b.norm(out, xp, wp, n)
		b.driver.MemCopyD2H(b.context, got, out)
		checkClose("rmsnorm", got, rmsCPU(x, w))
		b.norm(xp, xp, wp, n)
		b.driver.MemCopyD2H(b.context, got, xp)
		checkClose("rmsnorm in place", got, rmsCPU(x, w))
		b.driver.MemCopyH2D(b.context, xp, x)
		b.pair("kernel_swiglu", xp, wp, n)
		b.driver.MemCopyD2H(b.context, got, xp)
		activation := make([]float32, n)
		for i := range activation {
			activation[i] = x[i] / (1 + float32(math.Exp(float64(-x[i])))) * w[i]
		}
		checkClose("swiglu", got, activation)
		b.driver.MemCopyH2D(b.context, xp, x)
		b.pair("kernel_add", xp, wp, n)
		b.driver.MemCopyD2H(b.context, got, xp)
		want := make([]float32, n)
		for i := range want {
			want[i] = x[i] + w[i]
		}
		checkClose("add", got, want)
		matrix := make([]float32, n*3)
		for i := range matrix {
			matrix[i] = float32(i%7-3) * 0.01
		}
		mp, result := b.alloc(len(matrix)), b.alloc(3)
		b.driver.MemCopyH2D(b.context, mp, matrix)
		b.driver.MemCopyH2D(b.context, xp, x)
		b.mat(result, xp, mp, n, 3)
		matOut := make([]float32, 3)
		b.driver.MemCopyD2H(b.context, matOut, result)
		checkClose("matmul", matOut, matCPU(matrix, x, 3))
		for _, p := range []driver.Ptr{mp, result, xp, wp, out} {
			if err := b.driver.FreeMemory(b.context, p); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestGPUAttentionLongContext(t *testing.T) {
	path := os.Getenv("MGPUSIM_LLAMA_HSACO")
	platform := runner.MakeEmuBuilder().WithNumGPU(1).Build()
	b := NewBenchmark(platform.Driver)
	b.KernelPath = path
	if err := b.LoadKernels(); err != nil {
		t.Fatal(err)
	}
	platform.Driver.Run()
	defer platform.Engine.Finished()
	defer platform.Driver.Terminate()
	b.driver.SelectGPU(b.context, 1)
	const heads, hs, kv, seq = 4, 4, 8, 257
	q, k, v := make([]float32, heads*hs), make([]float32, seq*kv), make([]float32, seq*kv)
	for i := range q {
		q[i] = float32(i%7-3) * 0.2
	}
	for i := range k {
		k[i] = float32(i%11-5) * 0.1
		v[i] = float32(i%13-6) * 0.3
	}
	qp, kp, vp, att, out := b.alloc(len(q)), b.alloc(len(k)), b.alloc(len(v)), b.alloc(heads*seq), b.alloc(heads*hs)
	b.driver.MemCopyH2D(b.context, qp, q)
	b.driver.MemCopyH2D(b.context, kp, k)
	b.driver.MemCopyH2D(b.context, vp, v)
	for _, pos := range []int{0, 1, 129, 256} {
		b.launch("kernel_attention", heads*128, 128, &attentionArgs{Out: out, Q: qp, Keys: kp, Values: vp, Att: att, Head: hs, KV: kv, Mul: 2, Seq: seq, Pos: int32(pos)})
		got, want := make([]float32, heads*hs), make([]float32, heads*hs)
		b.driver.MemCopyD2H(b.context, got, out)
		for head := 0; head < heads; head++ {
			scores := make([]float32, pos+1)
			off := head / 2 * hs
			for p := range scores {
				for i := 0; i < hs; i++ {
					scores[p] += q[head*hs+i] * k[p*kv+off+i]
				}
				scores[p] /= 2
			}
			softCPU(scores)
			for i := 0; i < hs; i++ {
				for p, a := range scores {
					want[head*hs+i] += a * v[p*kv+off+i]
				}
			}
		}
		checkClose("long-context GQA", got, want)
	}
	for _, p := range []driver.Ptr{qp, kp, vp, att, out} {
		if err := b.driver.FreeMemory(b.context, p); err != nil {
			t.Fatal(err)
		}
	}
}
