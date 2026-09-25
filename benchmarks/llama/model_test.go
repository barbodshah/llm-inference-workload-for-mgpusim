// SPDX-License-Identifier: GPL-3.0-only
package llama

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func checkpointFixture(t *testing.T, shared bool) []byte {
	t.Helper()
	m := TinyModel()
	c := m.Config
	if !shared {
		c.VocabSize = -c.VocabSize
	}
	var buf bytes.Buffer
	put := func(x interface{}) {
		if err := binary.Write(&buf, binary.LittleEndian, x); err != nil {
			t.Fatal(err)
		}
	}
	put(c)
	for _, x := range [][]float32{m.Embedding, m.AttNorm, m.Q, m.K, m.V, m.O, m.FFNNorm, m.W1, m.W2, m.W3, m.FinalNorm} {
		put(x)
	}
	// Distinct sentinels detect incorrect skipping of the two legacy RoPE tables.
	tables := make([]float32, int(c.SeqLen*c.Dim/c.Heads))
	for i := range tables {
		tables[i] = 123
	}
	put(tables)
	if !shared {
		cls := append([]float32(nil), m.Embedding...)
		cls[0] = 42
		put(cls)
	}
	return buf.Bytes()
}

func TestCheckpointLayout(t *testing.T) {
	for _, shared := range []bool{true, false} {
		data := checkpointFixture(t, shared)
		m, err := readModel(bytes.NewReader(data), int64(len(data)))
		if err != nil {
			t.Fatal(err)
		}
		want := TinyModel()
		for _, pair := range [][2][]float32{{m.Embedding, want.Embedding}, {m.AttNorm, want.AttNorm}, {m.Q, want.Q}, {m.K, want.K}, {m.V, want.V}, {m.O, want.O}, {m.FFNNorm, want.FFNNorm}, {m.W1, want.W1}, {m.W2, want.W2}, {m.W3, want.W3}, {m.FinalNorm, want.FinalNorm}} {
			if len(pair[0]) != len(pair[1]) {
				t.Fatal("tensor size")
			}
			for i := range pair[0] {
				if pair[0][i] != pair[1][i] {
					t.Fatalf("tensor mismatch at %d", i)
				}
			}
		}
		if shared {
			if &m.Classifier[0] != &m.Embedding[0] {
				t.Fatal("shared classifier must alias embedding")
			}
		} else if m.Classifier[0] != 42 || &m.Classifier[0] == &m.Embedding[0] {
			t.Fatal("unshared classifier layout")
		}
	}
}

func TestRejectMalformedCheckpoint(t *testing.T) {
	original := checkpointFixture(t, true)
	for _, tc := range []struct {
		name   string
		offset int
		value  int32
	}{{"zero dimension", 0, 0}, {"odd head size", 0, 12}, {"invalid GQA", 16, 3}, {"overflow", 8, math.MaxInt32}, {"negative vocabulary overflow", 20, math.MinInt32}} {
		t.Run(tc.name, func(t *testing.T) {
			data := append([]byte(nil), original...)
			binary.LittleEndian.PutUint32(data[tc.offset:], uint32(tc.value))
			if _, err := readModel(bytes.NewReader(data), int64(len(data))); err == nil {
				t.Fatal("accepted invalid header")
			}
		})
	}
	for _, n := range []int{0, 27, len(original) - 4} {
		if _, err := readModel(bytes.NewReader(original[:n]), int64(n)); err == nil {
			t.Fatalf("accepted truncated size %d", n)
		}
	}
	if _, err := readModel(bytes.NewReader(original), int64(len(original)+4)); err == nil {
		t.Fatal("accepted trailing data")
	}
}

func TestScalarMath(t *testing.T) {
	got := matCPU([]float32{1, 2, 3, 4, 5, 6}, []float32{2, -1, 3}, 2)
	if got[0] != 9 || got[1] != 21 {
		t.Fatal(got)
	}
	x := []float32{1000, 1000}
	softCPU(x)
	if x[0] != 0.5 || x[1] != 0.5 {
		t.Fatal(x)
	}
	norm := rmsCPU([]float32{3, 4}, []float32{1, 2})
	if math.Abs(float64(norm[0])-3/math.Sqrt(12.50001)) > 1e-6 || math.Abs(float64(norm[1])-8/math.Sqrt(12.50001)) > 1e-6 {
		t.Fatal(norm)
	}
}

func TestReferenceAnalyticResidual(t *testing.T) {
	m := TinyModel()
	for _, w := range [][]float32{m.Q, m.K, m.V, m.O, m.W1, m.W2, m.W3} {
		clear(w)
	}
	ref := NewReference(m)
	for _, token := range []int{1, 2, 3} {
		out := ref.Forward(token)
		row := m.Embedding[token*16 : (token+1)*16]
		var sq float64
		for _, x := range row {
			sq += float64(x) * float64(x)
		}
		scale := 1 / math.Sqrt(sq/16+1e-5)
		for j := range out {
			var want float64
			for k, x := range row {
				want += float64(m.Classifier[j*16+k]) * float64(x) * scale
			}
			if math.Abs(float64(out[j])-want) > 1e-6 {
				t.Fatalf("logit %d got %g want %g", j, out[j], want)
			}
		}
	}
}

func TestReferenceCachePersists(t *testing.T) {
	m := TinyModel()
	a := NewReference(m)
	a.Forward(1)
	old := append([]float32(nil), a.keys...)
	out := a.Forward(2)
	for l := 0; l < int(m.Config.Layers); l++ {
		base := l * int(m.Config.SeqLen*m.Config.kvDim())
		for i := 0; i < int(m.Config.kvDim()); i++ {
			if a.keys[base+i] != old[base+i] {
				t.Fatal("overwrote previous position")
			}
		}
	}
	b := NewReference(m)
	b.Forward(3)
	other := b.Forward(2)
	different := false
	for i := range out {
		if out[i] != other[i] {
			different = true
		}
		if math.IsNaN(float64(out[i])) {
			t.Fatal("NaN")
		}
	}
	if !different {
		t.Fatal("history did not affect inference")
	}
	for a.next < int(m.Config.SeqLen) {
		a.Forward(1)
	}
	defer func() {
		if recover() == nil {
			t.Fatal("accepted context overflow")
		}
	}()
	a.Forward(1)
}

func TestVerificationRejectsNonfinite(t *testing.T) {
	for _, x := range []float32{float32(math.NaN()), float32(math.Inf(1)), 1} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("accepted bad output", x)
				}
			}()
			checkClose("test", []float32{x}, []float32{0})
		}()
	}
}
