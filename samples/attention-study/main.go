package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"sort"
	"strconv"
	"strings"

	"github.com/sarchlab/akita/v3/mem/mem"
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/akita/v3/tracing"
	"github.com/sarchlab/mgpusim/v3/benchmarks/attention"
	"github.com/sarchlab/mgpusim/v3/protocol"
	"github.com/sarchlab/mgpusim/v3/samples/runner"
	"github.com/sarchlab/mgpusim/v3/timing/cu"
	"github.com/sarchlab/mgpusim/v3/timing/wavefront"
)

type interval struct {
	kernel     uint64
	phase      string
	start, end float64
}

type measurements struct {
	waves                 map[string]float64
	waveSeconds           float64
	barriers              int
	clock                 sim.TimeTeller
	launches              map[string]*interval
	active                int
	readBytes, writeBytes uint64
}

func (m *measurements) StartTask(t tracing.Task) {
	if t.Kind == "wavefront" {
		m.waves[t.ID] = float64(m.clock.CurrentTime())
	}
	if t.Kind == "inst" && t.What == "Special" {
		inst := t.Detail.(map[string]interface{})["inst"].(*wavefront.Inst)
		if inst.Opcode == 10 {
			m.barriers++
		}
	}
	if t.Kind != "req_in" {
		return
	}
	switch req := t.Detail.(type) {
	case *protocol.LaunchKernelReq:
		phase := "decode"
		if req.Packet.WorkgroupSizeX == 256 {
			phase = "prefill"
		}
		m.launches[t.ID] = &interval{kernel: req.Packet.KernelObject, phase: phase, start: float64(m.clock.CurrentTime())}
		m.active++
	case *mem.ReadReq:
		if m.active > 0 {
			m.readBytes += req.GetByteSize()
		}
	case *mem.WriteReq:
		if m.active > 0 {
			m.writeBytes += req.GetByteSize()
		}
	}
}

func (*measurements) StepTask(tracing.Task) {}

func (m *measurements) EndTask(t tracing.Task) {
	if start, ok := m.waves[t.ID]; ok {
		m.waveSeconds += float64(m.clock.CurrentTime()) - start
		delete(m.waves, t.ID)
	}
	if run := m.launches[t.ID]; run != nil {
		run.end = float64(m.clock.CurrentTime())
		m.active--
	}
}

type edge struct {
	time   float64
	kernel uint64
	delta  int
}

func residencySeconds(records []attention.Residence) (active, mixed float64) {
	byCU := make(map[string][]edge)
	for _, r := range records {
		byCU[r.CU] = append(byCU[r.CU], edge{r.Start, r.Kernel, 1}, edge{r.End, r.Kernel, -1})
	}
	// Sum in a deterministic order.
	names := make([]string, 0, len(byCU))
	for name := range byCU {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		events := byCU[name]
		sort.Slice(events, func(i, j int) bool { return events[i].time < events[j].time })
		resident := make(map[uint64]int)
		last := events[0].time
		for _, e := range events {
			if len(resident) > 0 {
				active += e.time - last
			}
			if len(resident) > 1 {
				mixed += e.time - last
			}
			resident[e.kernel] += e.delta
			if resident[e.kernel] == 0 {
				delete(resident, e.kernel)
			}
			last = e.time
		}
	}
	return active, mixed
}

func main() {
	config := attention.DefaultConfig()
	flag.StringVar(&config.Phase, "phase", "both", "prefill, decode, or both")
	flag.StringVar(&config.Mode, "mode", "serial", "serial or concurrent")
	flag.IntVar(&config.Chunk, "chunk", 32, "prefill query tokens")
	flag.IntVar(&config.PrefillLength, "prefill-context", 96, "prefill KV tokens")
	batch := flag.Int("batch", 16, "number of decode requests")
	context := flag.Int("decode-context", 64, "KV tokens per decode request")
	contexts := flag.String("decode-contexts", "", "comma-separated per-request KV lengths; overrides batch and decode-context")
	lanes := flag.Int("simd-lanes", 16, "modeled SIMD FP32 lanes per cycle")
	column := flag.Int("column-cycles", 1, "minimum DRAM column command spacing")
	flag.BoolVar(&config.CompactDecode, "compact-decode", false, "use compact dimension-8 decode tiles")
	verify := flag.Bool("verify-outputs", false, "verify outputs using CPU reference with ordinary memory copies")
	flag.IntVar(&config.Heads, "heads", 8, "query heads")
	flag.IntVar(&config.KVHeads, "kv-heads", 8, "KV heads")
	flag.IntVar(&config.Dim, "head-dim", 8, "head dimension")
	prefillCUs := flag.Int("prefill-cus", 0, "reserve this many CUs for prefill and the remainder for decode; zero shares all CUs")
	flag.Parse()
	if *lanes < 1 || *lanes > 64 || 64%*lanes != 0 || *column < 1 {
		log.Fatal("invalid SIMD width or DRAM spacing")
	}

	if *contexts != "" {
		config.DecodeLengths = nil
		for _, value := range strings.Split(*contexts, ",") {
			length, err := strconv.Atoi(strings.TrimSpace(value))
			if err != nil {
				log.Fatal(err)
			}
			config.DecodeLengths = append(config.DecodeLengths, length)
		}
		*batch = len(config.DecodeLengths)
		if *batch > 128 {
			log.Fatal("decode-contexts supports at most 128 requests")
		}
	} else {
		if *batch < 1 || *batch > 128 {
			log.Fatal("batch must be 1..128")
		}
		config.DecodeLengths = make([]int, *batch)
		for i := range config.DecodeLengths {
			config.DecodeLengths[i] = *context
		}
	}
	totalContext := 0
	for _, length := range config.DecodeLengths {
		totalContext += length
	}
	if err := config.Validate(); err != nil {
		log.Fatal(err)
	}

	builder := runner.MakeR9NanoBuilder().WithNumGPU(1).WithDRAMColumnCycles(*column)
	if !*verify {
		builder = builder.WithMagicMemoryCopy()
	}
	platform := builder.Build()
	gpu := platform.GPUs[0]
	if *prefillCUs < 0 || *prefillCUs >= len(gpu.CUs) {
		log.Fatal("prefill-cus must be zero or smaller than total CUs")
	}
	if *prefillCUs > 0 {
		gpu.CommandProcessor.RestrictLaunchCU = func(req *protocol.LaunchKernelReq, index int) bool {
			prefill := req.Packet.WorkgroupSizeX == 256
			return prefill == (index < *prefillCUs)
		}
	}
	m := &measurements{clock: platform.Engine, launches: make(map[string]*interval), waves: make(map[string]float64)}
	tracing.CollectTrace(gpu.CommandProcessor, m)
	for _, controller := range gpu.MemControllers {
		tracing.CollectTrace(controller, m)
	}
	residency := attention.NewResidencyTracer(platform.Engine)
	stacks := make([]*cu.CPIStackTracer, 0, len(gpu.CUs))
	for _, unit := range gpu.CUs {
		tracing.CollectTrace(unit, residency)
		tracing.CollectTrace(unit, m)
		stack := cu.NewCPIStackInstHook(unit.(*cu.ComputeUnit), platform.Engine)
		tracing.CollectTrace(unit, stack)
		stacks = append(stacks, stack)
	}
	pipelines := make([]*tracing.BusyTimeTracer, 0, len(gpu.SIMDs))
	for _, unit := range gpu.SIMDs {
		unit.(*cu.SIMDUnit).NumSinglePrecisionUnit = *lanes
		tracer := tracing.NewBusyTimeTracer(platform.Engine, func(t tracing.Task) bool { return t.Kind == "pipeline" })
		tracing.CollectTrace(unit, tracer)
		pipelines = append(pipelines, tracer)
	}
	benchmark := attention.NewBenchmark(platform.Driver)
	benchmark.Config = config
	if *verify {
		benchmark.EnableVerification()
	}
	platform.Driver.Run()
	benchmark.Run()
	if *verify {
		benchmark.Verify()
	}
	platform.Driver.Terminate()
	platform.Engine.Finished()

	expected := 1
	if config.Phase == "both" {
		expected = 2
	}
	if len(m.launches) != expected || m.active != 0 || len(residency.Pending) != 0 || len(m.waves) != 0 {
		log.Fatalf("incomplete study: launches=%d active=%d pending workgroups=%d pending waves=%d",
			len(m.launches), m.active, len(residency.Pending), len(m.waves))
	}
	if *prefillCUs > 0 && residency.MixedCUCount() != 0 {
		log.Fatal("partitioned launches overlapped on a CU")
	}
	first, last := math.Inf(1), 0.0
	for _, run := range m.launches {
		first, last = math.Min(first, run.start), math.Max(last, run.end)
	}
	span := last - first
	var busy float64
	for _, p := range pipelines {
		busy += float64(p.BusyTime())
	}
	// Normalize per-CU CPI categories by that CU's total CPI, then average active CUs.
	shares := make(map[string]float64)
	count := 0
	for _, stack := range stacks {
		values := stack.GetCPIStack()
		total := values["total"]
		if math.IsNaN(total) || math.IsInf(total, 0) || total <= 0 {
			continue
		}
		count++
		for key, value := range values {
			if key != "total" {
				shares[key] += value / total
			}
		}
	}
	for key := range shares {
		shares[key] /= float64(count)
	}
	activeCUSeconds, mixedCUSeconds := residencySeconds(residency.Completed)
	result := map[string]any{
		"prefill_cus":                *prefillCUs,
		"head_dim":                   config.Dim,
		"heads":                      config.Heads,
		"kv_heads":                   config.KVHeads,
		"compact_decode":             config.CompactDecode,
		"mean_resident_waves_per_cu": m.waveSeconds / (span * float64(len(gpu.CUs))),
		"barrier_instructions":       m.barriers,
		"phase":                      config.Phase,
		"mode":                       config.Mode,
		"chunk":                      config.Chunk,
		"prefill_context":            config.PrefillLength,
		"batch":                      *batch,
		"decode_contexts":            config.DecodeLengths,
		"decode_kv_bytes":            totalContext * config.KVHeads * config.Dim * 8,
		"simd_lanes":                 *lanes,
		"column_cycles":              *column,
		"makespan_us":                span * 1e6,
		"dram_read_request_bytes":    m.readBytes, "dram_write_request_bytes": m.writeBytes,
		"simd_busy_fraction":  busy / (float64(len(pipelines)) * span),
		"active_cu_fraction":  activeCUSeconds / (span * float64(len(gpu.CUs))),
		"mixed_cus":           residency.MixedCUCount(),
		"mixed_cu_us":         mixedCUSeconds * 1e6,
		"cpi_activity_shares": shares,
	}
	if *contexts == "" {
		result["decode_context"] = *context
	}
	for _, run := range m.launches {
		var count int
		var admission, execution float64
		var lastGroup attention.Residence
		for _, group := range residency.Completed {
			if group.Kernel != run.kernel {
				continue
			}
			count++
			admission += group.Start - run.start
			execution += group.End - group.Start
			if group.End > lastGroup.End {
				lastGroup = group
			}
		}
		if count == 0 {
			log.Fatal("missing workgroups for ", run.phase)
		}
		result[run.phase+"_mean_pre_residency_us"] = 1e6 * admission / float64(count)
		result[run.phase+"_mean_residency_us"] = 1e6 * execution / float64(count)
		result[run.phase+"_last_group_pre_residency_us"] = 1e6 * (lastGroup.Start - run.start)
		result[run.phase+"_last_group_residency_us"] = 1e6 * (lastGroup.End - lastGroup.Start)
		result[run.phase+"_last_group_completion_tail_us"] = 1e6 * (run.end - lastGroup.End)
		result[run.phase+"_duration_us"] = (run.end - run.start) * 1e6
		result[run.phase+"_completion_us"] = (run.end - first) * 1e6
	}
	data, err := json.Marshal(result)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(string(data))
}
