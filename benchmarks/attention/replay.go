// SPDX-License-Identifier: MIT
package attention

import (
	"fmt"
	"math"
	"sort"

	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/akita/v3/tracing"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/insts"
	"github.com/sarchlab/mgpusim/v3/protocol"
)

// ReplayRequest describes an attention-only request with an explicit arrival time.
type ReplayRequest struct {
	ArrivalUS float64 `json:"arrival_us"`
	Prompt    int     `json:"prompt"`
	Output    int     `json:"output"`
}

type replayRequest struct {
	ReplayRequest
	prefilled, decoded int
	ready              sim.VTimeInSec
}

type ReplayLaunch struct {
	Phase             string    `json:"phase"`
	Requests          []int     `json:"requests"`
	Contexts          []int     `json:"contexts"`
	ReadyUS           []float64 `json:"ready_us"`
	IssueUS           float64   `json:"issue_us"`
	AdmissionUS       float64   `json:"cp_receive_us"`
	DoneUS            float64   `json:"done_us"`
	LastCTAWaitUS     float64   `json:"last_cta_wait_us"`
	LastCTAResidentUS float64   `json:"last_cta_resident_us"`
	kernel            uint64
	chunk             int
}

type ReplayResult struct {
	Policy           string          `json:"policy"`
	ElapsedUS        float64         `json:"elapsed_us"`
	Tokens           int             `json:"decode_tokens"`
	TokensPerSecond  float64         `json:"decode_tokens_per_attention_second"`
	P95FirstDecodeUS float64         `json:"p95_first_decode_us"`
	P95InterDecodeUS *float64        `json:"p95_inter_decode_us"`
	P95PrefillUS     float64         `json:"p95_prefill_us"`
	Launches         []*ReplayLaunch `json:"launches"`
}

type replayArrival struct {
	*sim.EventBase
	requests []int
}

type replay struct {
	*Benchmark
	engine                   sim.Engine
	programs                 map[string]*insts.HsaCo
	requests                 []replayRequest
	arrived                  []bool
	chunk                    int
	barrier                  bool
	prefill, decode, staging *driver.CommandQueue
	active                   map[string]*ReplayLaunch
	byKernel                 map[uint64]*ReplayLaunch
	prefillBusy, decodeBusy  bool
	first, gaps, prefills    []float64
	result                   ReplayResult
}

// RunReplay uses one serial timing engine and a driver with magic memory copies.
// The caller owns the platform and must leave the asynchronous driver stopped.
func (b *Benchmark) RunReplay(engine sim.Engine, cp tracing.NamedHookable, residency *ResidencyTracer,
	requests []ReplayRequest, chunk int, policy string) (ReplayResult, error) {
	if (policy != "barrier" && policy != "completion") || chunk < 1 || chunk > 4096 || len(requests) < 1 || len(requests) > 128 {
		return ReplayResult{}, fmt.Errorf("require barrier or completion, chunk 1..4096, and 1..128 requests")
	}
	programs, err := LoadKernels()
	if err != nil {
		return ReplayResult{}, err
	}
	r := &replay{Benchmark: b, engine: engine, programs: programs, chunk: chunk, barrier: policy == "barrier",
		arrived: make([]bool, len(requests)), active: make(map[string]*ReplayLaunch),
		byKernel: make(map[uint64]*ReplayLaunch), result: ReplayResult{Policy: policy}}
	b.Config = DefaultConfig()
	b.Config.Heads, b.Config.KVHeads = 8, 8
	b.Driver.SelectGPU(b.context, 1)
	r.prefill = b.Driver.CreateCommandQueue(b.context)
	r.decode = b.Driver.CreateCommandQueue(b.context)
	r.staging = b.Driver.CreateCommandQueue(b.context)
	firstArrival := math.Inf(1)
	wantTokens := 0
	maxContext := 0
	var prefillKV int64
	arrivals := make(map[sim.VTimeInSec][]int)
	for i, request := range requests {
		if math.IsNaN(request.ArrivalUS) || math.IsInf(request.ArrivalUS, 0) || request.ArrivalUS < 0 ||
			request.Prompt < 1 || request.Prompt > 4096 || request.Output < 1 || request.Output > 128 {
			return ReplayResult{}, fmt.Errorf("request %d has invalid arrival, prompt (1..4096), or output (1..128)", i)
		}
		firstArrival = math.Min(firstArrival, request.ArrivalUS)
		wantTokens += request.Output
		at := sim.VTimeInSec(request.ArrivalUS / 1e6)
		arrivals[at] = append(arrivals[at], i)
		r.requests = append(r.requests, replayRequest{ReplayRequest: request, ready: at})
		maxContext = max(maxContext, request.Prompt+request.Output)
		for prefix := 0; prefix < request.Prompt; prefix += chunk {
			prefillKV += int64(min(prefix+chunk, request.Prompt))
		}
	}
	// Each launch retains a distinct synthetic tensor allocation for this run.
	// Bound the padded KV footprint before scheduling any work.
	if (prefillKV+int64(wantTokens)*int64(maxContext))*512 > 256<<20 {
		return ReplayResult{}, fmt.Errorf("trace exceeds 256 MiB synthetic KV budget; reduce request counts or lengths")
	}
	for at, indices := range arrivals {
		engine.Schedule(replayArrival{sim.NewEventBase(at, r), indices})
	}
	tracing.CollectTrace(b.Driver, r)
	tracing.CollectTrace(cp, r)
	if err := engine.Run(); err != nil {
		return ReplayResult{}, err
	}
	if r.result.Tokens != wantTokens || len(r.active) != 0 || len(residency.Pending) != 0 {
		return ReplayResult{}, fmt.Errorf("replay stopped with unfinished work (%d of %d decode tokens)", r.result.Tokens, wantTokens)
	}
	lastGroup := make(map[uint64]Residence)
	for _, group := range residency.Completed {
		if group.End > lastGroup[group.Kernel].End {
			lastGroup[group.Kernel] = group
		}
	}
	for _, launch := range r.result.Launches {
		group, ok := lastGroup[launch.kernel]
		if !ok {
			return ReplayResult{}, fmt.Errorf("missing CTA residency for %s launch", launch.Phase)
		}
		launch.LastCTAWaitUS = float64(group.Start)*1e6 - launch.AdmissionUS
		launch.LastCTAResidentUS = float64(group.End-group.Start) * 1e6
		r.result.ElapsedUS = math.Max(r.result.ElapsedUS, launch.DoneUS-firstArrival)
	}
	r.result.TokensPerSecond = float64(r.result.Tokens) * 1e6 / r.result.ElapsedUS
	r.result.P95FirstDecodeUS = replayP95(r.first)
	r.result.P95PrefillUS = replayP95(r.prefills)
	if len(r.gaps) > 0 {
		p95 := replayP95(r.gaps)
		r.result.P95InterDecodeUS = &p95
	}
	return r.result, nil
}

func replayP95(values []float64) float64 {
	sort.Float64s(values)
	return values[int(math.Ceil(.95*float64(len(values))))-1]
}

func (r *replay) Handle(event sim.Event) error {
	for _, index := range event.(replayArrival).requests {
		r.arrived[index] = true
	}
	r.dispatch()
	return nil
}

func (r *replay) dispatch() {
	if r.barrier && (r.prefillBusy || r.decodeBusy) {
		return
	}
	now := float64(r.engine.CurrentTime()) * 1e6
	if !r.prefillBusy {
		// Oldest arrival first, with the trace index breaking ties.
		index := -1
		for i, request := range r.requests {
			if r.arrived[i] && request.prefilled < request.Prompt &&
				(index == -1 || request.ArrivalUS < r.requests[index].ArrivalUS) {
				index = i
			}
		}
		if index >= 0 {
			request := r.requests[index]
			chunk := min(r.chunk, request.Prompt-request.prefilled)
			r.launch(r.prefill, &ReplayLaunch{Phase: "prefill", Requests: []int{index},
				Contexts: []int{request.prefilled + chunk}, ReadyUS: []float64{float64(request.ready) * 1e6}, IssueUS: now, chunk: chunk})
			r.prefillBusy = true
		}
	}
	if !r.decodeBusy {
		job := &ReplayLaunch{Phase: "decode", IssueUS: now, chunk: 1}
		for i, request := range r.requests {
			if request.prefilled == request.Prompt && request.decoded < request.Output {
				job.Requests = append(job.Requests, i)
				job.Contexts = append(job.Contexts, request.Prompt+request.decoded+1)
				job.ReadyUS = append(job.ReadyUS, float64(request.ready)*1e6)
			}
		}
		if len(job.Requests) > 0 {
			r.launch(r.decode, job)
			r.decodeBusy = true
		}
	}
}

func (r *replay) launch(queue *driver.CommandQueue, job *ReplayLaunch) {
	d := makeDataset(job.chunk, r.Config.Heads, r.Config.KVHeads, r.Config.Dim, job.Contexts, 7)
	r.allocateDataset(d, func(dst driver.Ptr, src interface{}) {
		r.Driver.EnqueueMemCopyH2D(queue, dst, src)
	})
	qt, name := 1, "attention_decode_compact"
	if job.Phase == "prefill" {
		qt, name = 4, "attention_prefill"
	}
	groups := len(d.lengths) * d.heads * ((d.queries + qt - 1) / qt)
	r.Driver.EnqueueLaunchKernel(r.staging, r.programs[name], [3]uint32{uint32(groups * qt * 64), 1, 1},
		[3]uint16{uint16(qt * 64), 1, 1}, &d.args)
	for r.staging.NumCommand() > 0 {
		cmd := r.staging.Dequeue()
		if launch, ok := cmd.(*driver.LaunchKernelCommand); ok {
			job.kernel = launch.Packet.KernelObject
			r.byKernel[job.kernel] = job
			r.active[cmd.GetID()] = job
		}
		queue.Enqueue(cmd)
	}
	r.result.Launches = append(r.result.Launches, job)
	r.Driver.TickLater(r.engine.CurrentTime())
}

func (r *replay) StartTask(task tracing.Task) {
	req, ok := task.Detail.(*protocol.LaunchKernelReq)
	if ok && task.Kind == "req_in" {
		if job := r.byKernel[req.Packet.KernelObject]; job != nil {
			job.AdmissionUS = float64(r.engine.CurrentTime()) * 1e6
		}
	}
}

func (*replay) StepTask(tracing.Task) {}

func (r *replay) EndTask(task tracing.Task) {
	job := r.active[task.ID]
	if job == nil {
		return
	}
	delete(r.active, task.ID)
	now := r.engine.CurrentTime()
	job.DoneUS = float64(now) * 1e6
	if job.Phase == "prefill" {
		request := &r.requests[job.Requests[0]]
		request.prefilled += job.chunk
		request.ready = now
		if request.prefilled == request.Prompt {
			r.prefills = append(r.prefills, job.DoneUS-request.ArrivalUS)
		}
		r.prefillBusy = false
	} else {
		for _, i := range job.Requests {
			request := &r.requests[i]
			if request.decoded == 0 {
				r.first = append(r.first, job.DoneUS-request.ArrivalUS)
			} else {
				r.gaps = append(r.gaps, float64(now-request.ready)*1e6)
			}
			request.ready = now
			request.decoded++
			r.result.Tokens++
		}
		r.decodeBusy = false
	}
	r.dispatch()
}
