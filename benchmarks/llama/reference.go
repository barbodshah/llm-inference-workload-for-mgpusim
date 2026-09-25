// SPDX-License-Identifier: GPL-3.0-only
package llama

import "math"

// Reference is an independent scalar CPU forward pass with persistent KV state.
type Reference struct {
	model        *Model
	keys, values []float32
	next         int
}

func NewReference(m *Model) *Reference {
	n := int(m.Config.Layers) * int(m.Config.SeqLen) * int(m.Config.kvDim())
	return &Reference{model: m, keys: make([]float32, n), values: make([]float32, n)}
}
func rmsCPU(x, w []float32) []float32 {
	y := make([]float32, len(x))
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	scale := float32(1 / math.Sqrt(float64(ss/float32(len(x))+1e-5)))
	for i := range x {
		y[i] = w[i] * scale * x[i]
	}
	return y
}
func matCPU(w, x []float32, rows int) []float32 {
	y := make([]float32, rows)
	for i := range y {
		for j, v := range x {
			y[i] += w[i*len(x)+j] * v
		}
	}
	return y
}
func softCPU(x []float32) {
	mx := x[0]
	for _, v := range x {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i := range x {
		x[i] = float32(math.Exp(float64(x[i] - mx)))
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}

// Forward consumes the next token; positions must be contiguous from zero.
func (r *Reference) Forward(token int) []float32 {
	m := r.model
	c := m.Config
	pos := r.next
	if token < 0 || token >= int(c.VocabSize) || pos >= int(c.SeqLen) {
		panic("llama: token or position out of range")
	}
	d, h, kv, hs := int(c.Dim), int(c.HiddenDim), int(c.kvDim()), int(c.Dim/c.Heads)
	x := append([]float32(nil), m.Embedding[token*d:(token+1)*d]...)
	for l := 0; l < int(c.Layers); l++ {
		xb := rmsCPU(x, m.AttNorm[l*d:])
		q := matCPU(m.Q[l*d*d:], xb, d)
		k := matCPU(m.K[l*d*kv:], xb, kv)
		v := matCPU(m.V[l*d*kv:], xb, kv)
		for i := 0; i < d; i += 2 {
			freq := float32(1 / math.Pow(10000, float64(i%hs)/float64(hs)))
			angle := float32(pos) * freq
			co, si := float32(math.Cos(float64(angle))), float32(math.Sin(float64(angle)))
			a, b := q[i], q[i+1]
			q[i], q[i+1] = a*co-b*si, a*si+b*co
			if i < kv {
				a, b = k[i], k[i+1]
				k[i], k[i+1] = a*co-b*si, a*si+b*co
			}
		}
		base := l * int(c.SeqLen) * kv
		copy(r.keys[base+pos*kv:], k)
		copy(r.values[base+pos*kv:], v)
		xb = make([]float32, d)
		for head := 0; head < int(c.Heads); head++ {
			scores := make([]float32, pos+1)
			off := (head / int(c.Heads/c.KVHeads)) * hs
			for t := range scores {
				for i := 0; i < hs; i++ {
					scores[t] += q[head*hs+i] * r.keys[base+t*kv+off+i]
				}
				scores[t] /= float32(math.Sqrt(float64(hs)))
			}
			softCPU(scores)
			for i := 0; i < hs; i++ {
				for t, a := range scores {
					xb[head*hs+i] += a * r.values[base+t*kv+off+i]
				}
			}
		}
		y := matCPU(m.O[l*d*d:], xb, d)
		for i := range x {
			x[i] += y[i]
		}
		xb = rmsCPU(x, m.FFNNorm[l*d:])
		a := matCPU(m.W1[l*h*d:], xb, h)
		b := matCPU(m.W3[l*h*d:], xb, h)
		for i := range a {
			a[i] = a[i] / (1 + float32(math.Exp(float64(-a[i])))) * b[i]
		}
		y = matCPU(m.W2[l*d*h:], a, d)
		for i := range x {
			x[i] += y[i]
		}
	}
	r.next++
	return matCPU(m.Classifier, rmsCPU(x, m.FinalNorm), int(c.VocabSize))
}
