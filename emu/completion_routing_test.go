package emu

import (
	"github.com/sarchlab/akita/v3/sim"
	"github.com/sarchlab/mgpusim/v3/kernels"
	"github.com/sarchlab/mgpusim/v3/protocol"
	"testing"
)

type completionPort struct {
	sim.Port
	messages []sim.Msg
	failNext bool
}

func (p *completionPort) Send(msg sim.Msg) *sim.SendError {
	if p.failNext {
		p.failNext = false
		return sim.NewSendError()
	}
	p.messages = append(p.messages, msg)
	return nil
}

func TestFunctionalCompletionRouting(t *testing.T) {
	engine := sim.NewSerialEngine()
	port := &completionPort{}
	cu := &ComputeUnit{ToDispatcher: port, wfs: make(map[*kernels.WorkGroup][]*Wavefront)}
	cu.TickingComponent = sim.NewTickingComponent("TestCU", engine, sim.GHz, cu)
	a, b := sim.NewLimitNumMsgPort(nil, 1, "DispatcherA"), sim.NewLimitNumMsgPort(nil, 1, "DispatcherB")
	requests := []*protocol.MapWGReq{
		protocol.MapWGReqBuilder{}.WithSrc(a).WithWG(kernels.NewWorkGroup()).Build(),
		protocol.MapWGReqBuilder{}.WithSrc(b).WithWG(kernels.NewWorkGroup()).Build(),
	}
	for _, r := range requests {
		cu.wfs[r.WorkGroup] = nil
	}
	for _, r := range requests {
		if err := cu.handleWGCompleteEvent(NewWGCompleteEvent(0, cu, r)); err != nil {
			t.Fatal(err)
		}
	}
	if len(port.messages) != 2 {
		t.Fatalf("got %d replies, want one per dispatcher", len(port.messages))
	}
	for i, m := range port.messages {
		reply := m.(*protocol.WGCompletionMsg)
		if reply.Dst != requests[i].Src || len(reply.RspTo) != 1 || reply.RspTo[0] != requests[i].ID {
			t.Fatalf("misrouted completion %d: %+v", i, reply)
		}
	}
	if len(cu.wfs) != 0 {
		t.Fatal("workgroups not reclaimed")
	}
}

func TestFunctionalCompletionRetry(t *testing.T) {
	engine := sim.NewSerialEngine()
	port := &completionPort{failNext: true}
	cu := &ComputeUnit{ToDispatcher: port, wfs: make(map[*kernels.WorkGroup][]*Wavefront)}
	cu.TickingComponent = sim.NewTickingComponent("TestCU", engine, sim.GHz, cu)
	source := sim.NewLimitNumMsgPort(nil, 1, "Dispatcher")
	req := protocol.MapWGReqBuilder{}.WithSrc(source).WithWG(kernels.NewWorkGroup()).Build()
	cu.wfs[req.WorkGroup] = nil
	if err := cu.handleWGCompleteEvent(NewWGCompleteEvent(0, cu, req)); err != nil {
		t.Fatal(err)
	}
	if err := engine.Run(); err != nil {
		t.Fatal(err)
	}
	if len(port.messages) != 1 {
		t.Fatalf("got %d replies after retry", len(port.messages))
	}
	if port.messages[0].(*protocol.WGCompletionMsg).RspTo[0] != req.ID {
		t.Fatal("wrong retry ID")
	}
}
