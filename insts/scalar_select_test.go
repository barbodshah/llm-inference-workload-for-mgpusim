package insts_test

import (
	"github.com/sarchlab/mgpusim/v3/insts"
	"testing"
)

func TestDecodeScalarSelect64(t *testing.T) {
	// s_cselect_b64 s[4:5], s[0:1], s[2:3]
	inst, err := insts.NewDisassembler().Decode([]byte{0x00, 0x02, 0x84, 0x85})
	if err != nil {
		t.Fatal(err)
	}
	if inst.Dst.RegCount != 2 || inst.Src0.RegCount != 2 || inst.Src1.RegCount != 2 {
		t.Fatalf("64-bit select decoded with wrong register widths: %s", inst.String(nil))
	}
}
