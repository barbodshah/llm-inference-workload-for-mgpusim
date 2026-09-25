package emu

import (
	"github.com/sarchlab/mgpusim/v3/insts"
	"math"
	"testing"
)

func TestVOP3OrderedFloatComparisons(t *testing.T) {
	for _, tc := range []struct {
		opcode int
		name   string
		want   uint64
	}{{66, "v_cmp_eq_f32_e64", 0x19}, {70, "v_cmp_ge_f32_e64", 0x1b}} {
		state := &mockInstState{inst: insts.NewInst(), scratchpad: make([]byte, 4096)}
		state.inst.FormatType = insts.VOP3a
		state.inst.Opcode = insts.Opcode(tc.opcode)
		state.inst.InstName = tc.name
		sp := state.scratchpad.AsVOP3A()
		sp.EXEC = 0x3f
		sp.DST[0] = ^uint64(0)
		a := []float32{1, 2, -1, 0, float32(math.Inf(1)), float32(math.NaN()), 7}
		b := []float32{1, 1, 1, float32(math.Copysign(0, -1)), float32(math.Inf(1)), 0, 7}
		for i := range a {
			sp.SRC0[i] = uint64(math.Float32bits(a[i]))
			sp.SRC1[i] = uint64(math.Float32bits(b[i]))
		}
		NewALU(nil).Run(state)
		if sp.DST[0] != tc.want {
			t.Fatalf("%s mask=%x want=%x", tc.name, sp.DST[0], tc.want)
		}
		// The compiler uses absolute source modifiers in the RoPE math routines.
		sp.EXEC = 1
		sp.SRC0[0] = uint64(math.Float32bits(-2))
		sp.SRC1[0] = uint64(math.Float32bits(2))
		state.inst.Abs = 1
		NewALU(nil).Run(state)
		if sp.DST[0] != 1 {
			t.Fatal("absolute source modifier")
		}
	}
}

func TestUnsignedScalarBitExtract(t *testing.T) {
	for _, tc := range []struct {
		offset, width uint32
		want          uint64
	}{{0, 0, 0}, {0, 32, 0xfedcba98}, {4, 8, 0xa9}, {28, 16, 0xf}, {31, 1, 1}, {0, 127, 0xfedcba98}} {
		state := &mockInstState{inst: insts.NewInst(), scratchpad: make([]byte, 4096)}
		state.inst.FormatType = insts.SOP2
		state.inst.Opcode = 37
		sp := state.scratchpad.AsSOP2()
		sp.SRC0 = 0xfedcba98
		sp.SRC1 = uint64(tc.offset | tc.width<<16)
		sp.SCC = 1
		NewALU(nil).Run(state)
		wantSCC := byte(0)
		if tc.want != 0 {
			wantSCC = 1
		}
		if sp.DST != tc.want || sp.SCC != wantSCC {
			t.Fatalf("offset=%d width=%d got=%x SCC=%d", tc.offset, tc.width, sp.DST, sp.SCC)
		}
	}
}

func TestScalarSelect64(t *testing.T) {
	for _, scc := range []byte{0, 1} {
		state := &mockInstState{inst: insts.NewInst(), scratchpad: make([]byte, 4096)}
		state.inst.FormatType = insts.SOP2
		state.inst.Opcode = 11
		sp := state.scratchpad.AsSOP2()
		sp.SCC = scc
		sp.SRC0 = 0x123456789abcdef0
		sp.SRC1 = 0xfedcba9876543210
		NewALU(nil).Run(state)
		want := sp.SRC1
		if scc == 1 {
			want = sp.SRC0
		}
		if sp.DST != want || sp.SCC != scc {
			t.Fatal("64-bit select corrupted result or SCC")
		}
	}
}
