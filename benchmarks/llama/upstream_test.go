// SPDX-License-Identifier: GPL-3.0-only
package llama

import (
	"bytes"
	"encoding/binary"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func upstreamLogits(t *testing.T, executable, checkpoint string, vocab int) [][]float32 {
	t.Helper()
	cmd := exec.Command(executable, checkpoint, "1", "2", "3")
	var errors bytes.Buffer
	cmd.Stderr = &errors
	data, err := cmd.Output()
	if err != nil {
		t.Fatalf("upstream reference: %v: %s", err, errors.String())
	}
	if len(data) != 3*vocab*4 {
		t.Fatalf("upstream output: %d bytes, expected %d", len(data), 3*vocab*4)
	}
	result := make([][]float32, 3)
	r := bytes.NewReader(data)
	for i := range result {
		result[i] = make([]float32, vocab)
		if err := binary.Read(r, binary.LittleEndian, result[i]); err != nil {
			t.Fatal(err)
		}
	}
	return result
}

func TestUpstreamReference(t *testing.T) {
	executable := os.Getenv("MGPUSIM_LLAMA_REFERENCE")
	if executable == "" {
		t.Skip("set MGPUSIM_LLAMA_REFERENCE to compiled validation/reference.c")
	}
	for _, shared := range []bool{true, false} {
		path := filepath.Join(t.TempDir(), "tiny.bin")
		data := checkpointFixture(t, shared)
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		m, err := LoadModel(path)
		if err != nil {
			t.Fatal(err)
		}
		want := upstreamLogits(t, executable, path, int(m.Config.VocabSize))
		ref := NewReference(m)
		for i, token := range []int{1, 2, 3} {
			checkClose("upstream logits", ref.Forward(token), want[i])
		}
	}
	if path := os.Getenv("MGPUSIM_LLAMA_CHECKPOINT"); path != "" {
		m, err := LoadModel(path)
		if err != nil {
			t.Fatal(err)
		}
		want := upstreamLogits(t, executable, path, int(m.Config.VocabSize))
		ref := NewReference(m)
		for i, token := range []int{1, 2, 3} {
			checkClose("checkpoint upstream logits", ref.Forward(token), want[i])
		}
	}
}
