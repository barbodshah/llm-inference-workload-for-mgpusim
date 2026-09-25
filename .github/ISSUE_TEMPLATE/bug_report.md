---
name: Bug report
about: Report a reproducible workload, simulator, or correctness problem
title: ''
labels: bug
assignees: ''

---

**Environment**

- Repository commit:
- Operating system:
- Go version:
- Simulation mode: functional or timing
- Checkpoint: synthetic or checkpoint name and hash

**To reproduce**

Provide the smallest command that reproduces the problem:

```bash
go run ./samples/llama -verify
```

Include the token IDs and any non-default flags. Attach small text logs where
useful, but do not upload model checkpoints.

**Current behavior**

Describe what happened and include the complete error message.

**Expected behavior**

Describe the result you expected.

**Additional context**

Add any other information that may help reproduce the issue.
