$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$results = Join-Path $PSScriptRoot 'results'
New-Item -ItemType Directory -Force $results | Out-Null

function Invoke-Checked([scriptblock]$Action) {
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

Push-Location $repoRoot
try {
    Invoke-Checked { go test ./benchmarks/attention ./emu ./benchmarks/llama ./timing/cp/... ./timing/cu ./driver ./kernels ./insts -count=1 -v *> "$results/tests.txt" }
    Invoke-Checked { go vet ./benchmarks/attention ./samples/attention ./samples/attention-audit ./emu }
    $attention = Join-Path $results 'attention.exe'
    $audit = Join-Path $results 'attention-audit.exe'
    Invoke-Checked { go build -buildvcs=false -o $attention ./samples/attention }
    Invoke-Checked { go build -buildvcs=false -o $audit ./samples/attention-audit }
    Invoke-Checked { & $attention -verify -mode concurrent -chunk 5 -head-dim 7 -prefill-context 17 -decode-contexts '17,1,9' -metric-file-name "$results/functional-smoke" }
    foreach ($mode in @('same-queue', 'separate-queues')) {
        Invoke-Checked { & $audit -audit-mode $mode -residency-csv "$results/$mode.csv" }
    }
    $contexts = (@('128') * 32) -join ','
    foreach ($phase in @('prefill', 'decode')) {
        Invoke-Checked { & $attention -timing -report-all -phase $phase -heads 8 -kv-heads 2 -head-dim 32 -chunk 128 -prefill-context 128 -decode-contexts $contexts -metric-file-name "$results/profile-$phase" }
    }
    Invoke-Checked { python "$PSScriptRoot/summarize_profiles.py" }
    go version | Set-Content "$results/toolchain.txt"
    git rev-parse HEAD | Add-Content "$results/toolchain.txt"
    $sourcePaths = @(
        'benchmarks/attention/*.go', 'benchmarks/attention/kernels.cl',
        'benchmarks/attention/kernels.hsaco', 'benchmarks/attention/build-kernels.sh',
        'samples/attention/*.go', 'samples/attention-audit/*.go',
        'emu/computeunit.go', 'emu/completion_routing_test.go'
    )
    Get-ChildItem -Path $sourcePaths -File | Sort-Object FullName |
        ForEach-Object {
            $relative = [IO.Path]::GetRelativePath($repoRoot, $_.FullName)
            '{0}  {1}' -f (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant(), $relative
        } | Set-Content "$results/source-sha256.txt"
} finally {
    Pop-Location
}
