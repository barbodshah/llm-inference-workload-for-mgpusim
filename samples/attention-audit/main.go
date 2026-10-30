package main

import (
	"flag"
	"fmt"
	"github.com/sarchlab/mgpusim/v3/benchmarks/attention"
	"log"
	"os"
)

func main() {
	mode := flag.String("audit-mode", "separate-queues", "same-queue or separate-queues")
	groups := flag.Int("probe-groups", 4, "Workgroups per kernel, 1..256")
	rounds := flag.Int("probe-rounds", 512, "Dependent integer iterations per lane")
	path := flag.String("residency-csv", "", "Optional workgroup residency trace")
	flag.Parse()
	result, err := attention.RunConcurrencyAudit(*mode, *groups, *rounds)
	auditErr := err
	if result.Residency == nil {
		log.Fatal(auditErr)
	}
	if *path != "" {
		f, err := os.Create(*path)
		if err != nil {
			log.Fatal(err)
		}
		if err = result.Residency.WriteCSV(f); err != nil {
			log.Fatal(err)
		}
		if err = f.Close(); err != nil {
			log.Fatal(err)
		}
	}
	fmt.Printf("mode=%s verified_launches=%d completed_workgroups=%d mixed_CUs=%d pending=%d\n", result.Mode, result.Launches, result.Workgroups, result.MixedCUs, len(result.Residency.Pending))
	if auditErr != nil {
		log.Fatal(auditErr)
	}
}
