// SPDX-License-Identifier: GPL-3.0-only
package llama

import (
	"encoding/csv"
	"os"
	"strconv"
)

// PhaseMetric separates simulator time from the real time spent simulating.
// Forward includes embedding copy, kernel dispatches, and logits readback.
type PhaseMetric struct {
	Phase                         string
	Position                      int
	SimulatedSeconds, WallSeconds float64
}

func (b *Benchmark) writePhases() error {
	f, err := os.Create(b.PhaseMetricsPath)
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	err = w.Write([]string{"phase", "position", "simulated_seconds", "wall_seconds"})
	for _, p := range b.Phases {
		if err != nil {
			break
		}
		err = w.Write([]string{p.Phase, strconv.Itoa(p.Position), strconv.FormatFloat(p.SimulatedSeconds, 'g', 17, 64), strconv.FormatFloat(p.WallSeconds, 'g', 17, 64)})
	}
	w.Flush()
	if err == nil {
		err = w.Error()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}
