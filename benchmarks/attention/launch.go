package attention

import (
	"fmt"
	"github.com/sarchlab/mgpusim/v3/driver"
	"github.com/sarchlab/mgpusim/v3/insts"
)

// prepareLaunch uploads code, arguments, and dispatch packet before exposing
// the launch command to an execution queue. EnqueueLaunchKernel normally puts
// three uploads ahead of the launch. Those uploads can skew short-kernel
// concurrency tests and trigger context-wide cache flushing after another
// kernel starts. This helper uses only the existing driver queue API.
func prepareLaunch(d *driver.Driver, ctx *driver.Context, co *insts.HsaCo, grid [3]uint32, group [3]uint16, args interface{}) driver.Command {
	staging := d.CreateCommandQueue(ctx)
	d.EnqueueLaunchKernel(staging, co, grid, group, args)
	uploads := d.CreateCommandQueue(ctx)
	var launch driver.Command
	for staging.NumCommand() > 0 {
		cmd := staging.Dequeue()
		if _, ok := cmd.(*driver.LaunchKernelCommand); ok {
			if launch != nil {
				panic("multiple launch commands")
			}
			launch = cmd
		} else {
			uploads.Enqueue(cmd)
		}
	}
	if launch == nil {
		panic(fmt.Errorf("launch preparation produced no launch command"))
	}
	d.DrainCommandQueue(uploads)
	return launch
}
