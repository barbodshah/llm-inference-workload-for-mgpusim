// SPDX-License-Identifier: GPL-3.0-only
// Package llama implements sequential FP32 Llama inference for MGPUSim.
package llama

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// Config describes the legacy llama2.c checkpoint header.
type Config struct{ Dim, HiddenDim, Layers, Heads, KVHeads, VocabSize, SeqLen int32 }

func (c Config) validate() error {
	for _, n := range []int32{c.Dim, c.HiddenDim, c.Layers, c.Heads, c.KVHeads, c.VocabSize, c.SeqLen} {
		if n <= 0 {
			return fmt.Errorf("all model dimensions must be positive")
		}
	}
	if c.Dim%c.Heads != 0 || c.Heads%c.KVHeads != 0 || (c.Dim/c.Heads)%2 != 0 {
		return fmt.Errorf("invalid head dimensions: require dim divisible by heads, heads divisible by KV heads, and even head size")
	}
	// Kernels use signed 32-bit tensor indices. Bound every tensor and cache.
	for _, dims := range [][]int32{{c.Layers, c.Dim, c.Dim}, {c.Layers, c.Dim, c.HiddenDim}, {c.Layers, c.SeqLen, c.kvDim()}, {c.VocabSize, c.Dim}, {c.Heads, c.SeqLen}, {c.SeqLen, c.Dim / c.Heads}} {
		n := int64(1)
		for _, d := range dims {
			if n > math.MaxInt32/int64(d) {
				return fmt.Errorf("tensor exceeds 32-bit kernel indexing")
			}
			n *= int64(d)
		}
	}
	return nil
}
func (c Config) kvDim() int32 { return (c.Dim / c.Heads) * c.KVHeads }

// Model holds row-major weights. Classifier may alias Embedding.
type Model struct {
	Config                                                                     Config
	Embedding, AttNorm, Q, K, V, O, FFNNorm, W1, W2, W3, FinalNorm, Classifier []float32
}

func (m *Model) validate() error {
	if m == nil {
		return fmt.Errorf("missing model")
	}
	if err := m.Config.validate(); err != nil {
		return err
	}
	c := m.Config
	d, h, l, k, v := int(c.Dim), int(c.HiddenDim), int(c.Layers), int(c.kvDim()), int(c.VocabSize)
	for _, t := range []struct {
		name string
		data []float32
		n    int
	}{{"embedding", m.Embedding, v * d}, {"attention norm", m.AttNorm, l * d}, {"Q", m.Q, l * d * d}, {"K", m.K, l * k * d}, {"V", m.V, l * k * d}, {"O", m.O, l * d * d}, {"FFN norm", m.FFNNorm, l * d}, {"W1", m.W1, l * h * d}, {"W2", m.W2, l * d * h}, {"W3", m.W3, l * h * d}, {"final norm", m.FinalNorm, d}, {"classifier", m.Classifier, v * d}} {
		if len(t.data) != t.n {
			return fmt.Errorf("%s has %d elements; expected %d", t.name, len(t.data), t.n)
		}
	}
	return nil
}

// LoadModel reads the original unquantized llama2.c .bin format, not GGUF.
func LoadModel(path string) (*Model, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	return readModel(f, info.Size())
}

func readModel(r io.Reader, size int64) (*Model, error) {
	var c Config
	if err := binary.Read(r, binary.LittleEndian, &c); err != nil {
		return nil, fmt.Errorf("checkpoint header: %w", err)
	}
	shared := c.VocabSize > 0
	if c.VocabSize == math.MinInt32 {
		return nil, fmt.Errorf("invalid vocabulary size")
	}
	if c.VocabSize < 0 {
		c.VocabSize = -c.VocabSize
	}
	if err := c.validate(); err != nil {
		return nil, err
	}
	d, h, l, k, v, s := int64(c.Dim), int64(c.HiddenDim), int64(c.Layers), int64(c.kvDim()), int64(c.VocabSize), int64(c.SeqLen)
	counts := []int64{v * d, l * d, l * d * d, l * k * d, l * k * d, l * d * d, l * d, l * h * d, l * d * h, l * h * d, d, s * (d / int64(c.Heads))}
	total := int64(0)
	for _, n := range counts {
		total += n
	}
	if !shared {
		total += v * d
	}
	if size != 28+4*total {
		return nil, fmt.Errorf("checkpoint size: got %d bytes, expected %d (legacy FP32 format)", size, 28+4*total)
	}
	m := &Model{Config: c}
	tensors := []*[]float32{&m.Embedding, &m.AttNorm, &m.Q, &m.K, &m.V, &m.O, &m.FFNNorm, &m.W1, &m.W2, &m.W3, &m.FinalNorm}
	for i, t := range tensors {
		*t = make([]float32, int(counts[i]))
		if err := binary.Read(r, binary.LittleEndian, *t); err != nil {
			return nil, err
		}
	}
	if _, err := io.CopyN(io.Discard, r, 4*counts[11]); err != nil {
		return nil, err
	}
	if shared {
		m.Classifier = m.Embedding
	} else {
		m.Classifier = make([]float32, int(v*d))
		if err := binary.Read(r, binary.LittleEndian, m.Classifier); err != nil {
			return nil, err
		}
	}
	return m, nil
}

// TinyModel is a deterministic two-layer GQA fixture, not a trained model.
func TinyModel() *Model {
	c := Config{Dim: 16, HiddenDim: 24, Layers: 2, Heads: 4, KVHeads: 2, VocabSize: 32, SeqLen: 8}
	m := &Model{Config: c}
	d, h, l, k, v := int(c.Dim), int(c.HiddenDim), int(c.Layers), int(c.kvDim()), int(c.VocabSize)
	seed := uint32(1)
	tensor := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			seed = 1664525*seed + 1013904223
			x[i] = (float32(seed>>8)/16777216 - 0.5) * 0.2
		}
		return x
	}
	norm := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = 1
		}
		return x
	}
	m.Embedding = tensor(v * d)
	m.AttNorm = norm(l * d)
	m.Q = tensor(l * d * d)
	m.K = tensor(l * k * d)
	m.V = tensor(l * k * d)
	m.O = tensor(l * d * d)
	m.FFNNorm = norm(l * d)
	m.W1 = tensor(l * h * d)
	m.W2 = tensor(l * d * h)
	m.W3 = tensor(l * h * d)
	m.FinalNorm = norm(d)
	m.Classifier = m.Embedding
	return m
}
