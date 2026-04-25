$ErrorActionPreference = "Stop"

$RL_SERVER = "http://127.0.0.1:8000"
$CASES = @("stable_low", "stable_high", "gradual_ramp", "sudden_spike")
$MODES = @("inactive", "active")

if (-not (Test-Path results)) { New-Item -ItemType Directory -Force -Path results | Out-Null }
if (-not (Test-Path checkpoints)) { New-Item -ItemType Directory -Force -Path checkpoints | Out-Null }

pip install locust --quiet

function Get-Steps {
    try {
        $res = Invoke-RestMethod -Uri "$RL_SERVER/status" -Method Get
        return $res.steps
    } catch { return 0 }
}

function Set-Mode($mode) {
    try { Invoke-RestMethod -Uri "$RL_SERVER/set-mode?mode=$mode" -Method Post | Out-Null } catch {}
}

function Set-Learning($enabled) {
    if ($enabled) { $bool = "true" } else { $bool = "false" }
    try { Invoke-RestMethod -Uri "$RL_SERVER/set-learning?enabled=$bool" -Method Post | Out-Null } catch {}
}

function Save-Chart($mode, $case_name, $start_step, $end_step) {
    if (-not $start_step) { $start_step = 0 }
    if (-not $end_step) { $end_step = 0 }
    $out = "checkpoints/chart_$($case_name)_$($mode).png"
    $title = "Case=$case_name Mode=$mode Steps=$start_step-$end_step"
    try { 
        Invoke-RestMethod -Uri "$RL_SERVER/save-metrics?start_step=$start_step&end_step=$end_step&output=$out&title=$title" -Method Post | Out-Null 
    } catch {}
    return $out
}

Write-Host "Resetting in-memory metrics..."
try { 
    Invoke-RestMethod -Uri "$RL_SERVER/reset-metrics" -Method Post | Out-Null 
} catch {
    Write-Host "RL Server is NOT running at $RL_SERVER! Please check Terminal 1." -ForegroundColor Red
    exit 1
}

Write-Host "Disabling online learning for fair matrix comparison..."
Set-Learning $false

"mode,case,start_step,end_step,chart_path" | Out-File -FilePath "results/case_ranges.csv" -Encoding UTF8

foreach ($mode in $MODES) {
    Write-Host "=== MODE: $mode ===" -ForegroundColor Cyan
    Set-Mode $mode
    foreach ($case in $CASES) {
        Write-Host "--- CASE: $case ---" -ForegroundColor Yellow
        $start_step = Get-Steps
        
        $env:LOAD_CASE = $case
        .\run_loadtest.ps1
        
        $end_step = Get-Steps
        $chart_path = Save-Chart $mode $case $start_step $end_step
        "$mode,$case,$start_step,$end_step,$chart_path" | Out-File -FilePath "results/case_ranges.csv" -Append -Encoding UTF8
        Write-Host "Saved $chart_path for step range $start_step-$end_step`n" -ForegroundColor Green
    }
}

Set-Mode "active"
Set-Learning $true
Write-Host "All runs complete. Metadata: locust/results/case_ranges.csv" -ForegroundColor Cyan
