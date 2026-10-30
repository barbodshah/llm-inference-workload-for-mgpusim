package attention

import (
	"encoding/csv"
	"fmt"
	"io"
	"strconv"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/akita/v3/tracing"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/protocol"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
)

type Residence struct {
	CU, Workgroup string
	Kernel        uint64
	Start, End    float64
}

// ResidencyTracer observes existing MapWGReq receipt/completion traces. It does
// not change scheduling. Only use it with the serial simulation engine.
type ResidencyTracer struct {
	Clock     sim.TimeTeller
	Pending   map[string]Residence
	Completed []Residence
}

func NewResidencyTracer(clock sim.TimeTeller) *ResidencyTracer {
	return &ResidencyTracer{Clock: clock, Pending: make(map[string]Residence)}
}
func (t *ResidencyTracer) StartTask(task tracing.Task) {
	req, ok := task.Detail.(*protocol.MapWGReq)
	if !ok || task.Kind != "req_in" {
		return
	}
	t.Pending[task.ID] = Residence{CU: task.Where, Workgroup: req.WorkGroup.UID, Kernel: req.WorkGroup.Packet.KernelObject, Start: float64(t.Clock.CurrentTime())}
}
func (t *ResidencyTracer) StepTask(task tracing.Task) {}
func (t *ResidencyTracer) EndTask(task tracing.Task) {
	r, ok := t.Pending[task.ID]
	if !ok {
		return
	}
	r.End = float64(t.Clock.CurrentTime())
	t.Completed = append(t.Completed, r)
	delete(t.Pending, task.ID)
}

// MixedCUCount counts CUs with strict temporal overlap of workgroups belonging
// to distinct kernel allocations. Residency is not a measure of useful overlap.
func (t *ResidencyTracer) MixedCUCount() int {
	mixed := make(map[string]bool)
	for i, a := range t.Completed {
		for _, b := range t.Completed[i+1:] {
			if a.CU == b.CU && a.Kernel != b.Kernel && a.Start < b.End && b.Start < a.End {
				mixed[a.CU] = true
			}
		}
	}
	return len(mixed)
}
func (t *ResidencyTracer) WriteCSV(w io.Writer) error {
	c := csv.NewWriter(w)
	if err := c.Write([]string{"cu", "workgroup", "kernel_object", "start_s", "end_s"}); err != nil {
		return err
	}
	for _, r := range t.Completed {
		if err := c.Write([]string{r.CU, r.Workgroup, strconv.FormatUint(r.Kernel, 10), fmt.Sprintf("%.12g", r.Start), fmt.Sprintf("%.12g", r.End)}); err != nil {
			return err
		}
	}
	c.Flush()
	return c.Error()
}

type AuditResult struct {
	Mode                           string
	Launches, Workgroups, MixedCUs int
	Residency                      *ResidencyTracer
}

// RunConcurrencyAudit launches two probe kernels twice on a fresh, unmodified
// detailed platform. Same-queue execution is the ordering negative control.
func RunConcurrencyAudit(mode string, groups, rounds int) (AuditResult, error) {
	result := AuditResult{Mode: mode}
	if (mode != "same-queue" && mode != "separate-queues") || groups < 1 || groups > 256 || rounds < 1 || rounds > 65536 {
		return result, fmt.Errorf("invalid audit configuration")
	}
	programs, err := LoadKernels()
	if err != nil {
		return result, err
	}
	p := runner.MakeR9NanoBuilder().WithNumGPU(1).Build()
	tracer := NewResidencyTracer(p.Engine)
	result.Residency = tracer
	for _, cu := range p.GPUs[0].CUs {
		tracing.CollectTrace(cu, tracer)
	}
	p.Driver.Run()
	defer p.Engine.Finished()
	defer p.Driver.Terminate()
	ctx := p.Driver.Init()
	p.Driver.SelectGPU(ctx, 1)
	queues := []*driver.CommandQueue{p.Driver.CreateCommandQueue(ctx), p.Driver.CreateCommandQueue(ctx)}
	if mode == "same-queue" {
		queues[1] = queues[0]
	}
	outputs := []driver.Ptr{p.Driver.AllocateMemory(ctx, uint64(groups*64*4)), p.Driver.AllocateMemory(ctx, uint64(groups*64*4))}
	defer func() {
		for _, ptr := range outputs {
			_ = p.Driver.FreeMemory(ctx, ptr)
		}
	}()
	for repeat := 0; repeat < 2; repeat++ {
		seeds := []uint32{uint32(11 + repeat), uint32(97 + repeat)}
		launches := make([]driver.Command, 2)
		for i := range queues {
			args := probeArgs{Out: outputs[i], Rounds: uint32(rounds), Seed: seeds[i]}
			launches[i] = prepareLaunch(p.Driver, ctx, programs["concurrency_probe"], [3]uint32{uint32(groups * 64), 1, 1}, [3]uint16{64, 1, 1}, &args)
		}
		for i, cmd := range launches {
			queues[i].Enqueue(cmd)
			result.Launches++
		}
		for _, q := range queues {
			p.Driver.DrainCommandQueue(q)
		}
		for i, ptr := range outputs {
			got := make([]uint32, groups*64)
			p.Driver.MemCopyD2H(ctx, got, ptr)
			for j, x := range got {
				want := seeds[i] + uint32(j)
				for k := 0; k < rounds; k++ {
					want = want*1664525 + 1013904223
				}
				if x != want {
					return result, fmt.Errorf("probe repeat %d output %d index %d: %d != %d", repeat, i, j, x, want)
				}
			}
		}
		if len(tracer.Pending) != 0 {
			return result, fmt.Errorf("%d workgroups missing completion", len(tracer.Pending))
		}
	}
	result.Workgroups = len(tracer.Completed)
	result.MixedCUs = tracer.MixedCUCount()
	if result.Workgroups != 4*groups {
		return result, fmt.Errorf("completed %d workgroups, expected %d", result.Workgroups, 4*groups)
	}
	if mode == "same-queue" && result.MixedCUs != 0 {
		return result, fmt.Errorf("same-queue kernels overlapped")
	}
	if mode == "separate-queues" && result.MixedCUs == 0 {
		return result, fmt.Errorf("no same-CU overlap observed; increase probe duration or investigate launch path")
	}
	return result, nil
}
