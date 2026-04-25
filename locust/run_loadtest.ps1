$ErrorActionPreference = "Stop"

if (-not (Get-Command locust -ErrorAction SilentlyContinue)) {
    Write-Host "Installing locust..."
    pip install locust --quiet
}

if (-not $env:OPENFAAS_USER) { $env:OPENFAAS_USER="admin" }
if (-not $env:OPENFAAS_PASS) { $env:OPENFAAS_PASS="admin" }

try {
    $secret = kubectl get secret -n openfaas basic-auth -o jsonpath="{.data.basic-auth-password}" 2>$null
    if ($secret) {
        $env:OPENFAAS_PASS = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($secret))
    }
} catch {}

if (-not $env:FAAS_FUNCTION) { $env:FAAS_FUNCTION="figlet-fn" }
if (-not $env:LOAD_CASE) { $env:LOAD_CASE="stable_low" }
if (-not $env:OPENFAAS_URL) { $env:OPENFAAS_URL="http://127.0.0.1:8080" }

Write-Host "OPENFAAS_USER: $($env:OPENFAAS_USER)"
Write-Host "FAAS_FUNCTION: $($env:FAAS_FUNCTION)"
Write-Host "Gateway: $($env:OPENFAAS_URL)"
Write-Host "LOAD_CASE: $($env:LOAD_CASE)"
Write-Host ""

function Reset-Replicas {
    Write-Host "Resetting replicas to 1 for $($env:FAAS_FUNCTION)..."
    $pair = "$($env:OPENFAAS_USER):$($env:OPENFAAS_PASS)"
    $bytes = [System.Text.Encoding]::ASCII.GetBytes($pair)
    $base64 = [System.Convert]::ToBase64String($bytes)
    $headers = @{ Authorization = "Basic $base64" }

    try {
        Invoke-RestMethod -Uri "$($env:OPENFAAS_URL)/system/scale-function/$($env:FAAS_FUNCTION)" -Method Post -Headers $headers -Body '{"service":"'"$env:FAAS_FUNCTION"'","replicas":1}' -ContentType "application/json" | Out-Null
    } catch {}
    
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $resp = Invoke-RestMethod -Uri "$($env:OPENFAAS_URL)/system/function/$($env:FAAS_FUNCTION)" -Method Get -Headers $headers -ErrorAction Stop
            if ($resp.availableReplicas -eq 1) {
                Write-Host "Replicas confirmed at 1"
                return $true
            }
        } catch {}
        Start-Sleep -Seconds 1
    }
    Write-Host "Failed to confirm replicas=1 before test start"
}

Reset-Replicas
Write-Host "Starting headless load test..."
locust -f locustfile.py --host "$($env:OPENFAAS_URL)" --headless --users 200 --spawn-rate 20 --run-time 250s --html "report_$($env:LOAD_CASE).html" --exit-code-on-error 0
Write-Host "`nDone. Report saved to locust/report_$($env:LOAD_CASE).html"
