# Ridge train-window sweep: h5/h10/h20 x (6,12 | 12,24) months
$ErrorActionPreference = "Continue"
$Root = "F:\PythonProject\quant_trading"
Set-Location $Root

$env:TRAIN_N_JOBS = "20"
$env:TRAIN_MAX_WORKERS = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

$Py = Join-Path $Root ".venv\Scripts\python.exe"
$OutDir = Join-Path $Root "results\ridge_window_sweep"
$FactorCfg = Join-Path $Root "config\factor_configs.yaml"
$Log = Join-Path $OutDir "sweep.log"

$runs = @(
    @{ Horizon = 5;  Windows = "6,12" },
    @{ Horizon = 5;  Windows = "12,24" },
    @{ Horizon = 10; Windows = "6,12" },
    @{ Horizon = 10; Windows = "12,24" },
    @{ Horizon = 20; Windows = "6,12" },
    @{ Horizon = 20; Windows = "12,24" }
)

$builtHorizon = @{}
$failures = @()
$i = 0
foreach ($r in $runs) {
    $i++
    $tag = "ridge_h$($r.Horizon)_w$($r.Windows -replace ',','-')"
    $metrics = Join-Path $OutDir "model_metrics_$tag.json"
    if (Test-Path $metrics) {
        "SKIP existing $tag $(Get-Date -Format o)" | Tee-Object -FilePath $Log -Append
        if (-not $builtHorizon.ContainsKey($r.Horizon)) { $builtHorizon[$r.Horizon] = $true }
        continue
    }
    $skipFactor = @()
    if ($builtHorizon.ContainsKey($r.Horizon)) { $skipFactor = @("--skip-factor-build") }
    "===== Run $i/6: horizon=$($r.Horizon) train-windows=$($r.Windows) =====" | Tee-Object -FilePath $Log -Append
    $args = @(
        "-u", "run.py",
        "--skip-download",
        "--mode", "ridge",
        "--horizon", "$($r.Horizon)",
        "--train-windows", $r.Windows,
        "--factor-config", $FactorCfg,
        "--output-dir", $OutDir
    ) + $skipFactor
    & $Py @args 2>&1 | Tee-Object -FilePath $Log -Append
    if ($LASTEXITCODE -ne 0) {
        $msg = "Run failed: horizon=$($r.Horizon) windows=$($r.Windows) exit=$LASTEXITCODE"
        $msg | Tee-Object -FilePath $Log -Append
        $failures += $msg
    } else {
        $builtHorizon[$r.Horizon] = $true
    }
}
if ($failures.Count -eq 0) {
    "===== All 6 ridge window sweep runs completed $(Get-Date -Format o) =====" | Tee-Object -FilePath $Log -Append
    exit 0
} else {
    "===== Sweep finished with $($failures.Count) failure(s) $(Get-Date -Format o) =====" | Tee-Object -FilePath $Log -Append
    $failures | Tee-Object -FilePath $Log -Append
    exit 1
}
