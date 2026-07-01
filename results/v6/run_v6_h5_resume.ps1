# v6 h5 续跑 — 仅补未完成模型（默认 dynamic）；无 linear/rf/mlp
# 因子面板：首个需算因子的 run 自动写 data/processed/factor_panel_*.parquet，后续同 horizon+YAML 自动命中缓存
# IC/YAML 变更后首个 run 加 --rebuild-factor-cache
$ErrorActionPreference = "Continue"
Set-Location "F:\PythonProject\quant_trading"

chcp 65001 | Out-Null
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding  = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)

$env:LOGURU_COLORIZE  = "0"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8       = "1"
$env:PYTHONUNBUFFERED = "1"
$env:TRAIN_N_JOBS = "10"
$env:DYNAMIC_MAX_WORKERS = "4"

$py = ".venv\Scripts\python.exe"
$yaml = "config\factor_configs.yaml"
$out = "results\v6"
$logDir = "$out\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Test-ModeDone([string]$mode) {
    $tag = "${mode}_h5"
    $metrics = Join-Path $out "model_metrics_${tag}.json"
    $backtest = Join-Path $out "backtest_${tag}.png"
    if ($mode -eq "dynamic") {
        return (Test-Path $backtest) -and (Test-Path (Join-Path $out "factor_scores_${tag}.parquet"))
    }
    return (Test-Path $metrics) -and (Test-Path $backtest)
}

# Done: ridge,lgbm,xgb,cat. h5 skips rf/mlp; only dynamic remains.
$modes = @('dynamic')
$driverLog = "$logDir\driver_v6_h5_resume.log"
"=== v6 h5 RESUME $(Get-Date -Format o) modes=$($modes -join ',') ===" | Tee-Object -Append $driverLog

foreach ($mode in $modes) {
    if (Test-ModeDone $mode) {
        "SKIP $mode h5 (already done)" | Tee-Object -Append $driverLog
        continue
    }
    $runLog = "$logDir\run_${mode}_h5.log"
    "START $mode h5 $(Get-Date -Format o)" | Tee-Object -Append $driverLog
    & $py -u run.py --skip-download --mode $mode --horizon 5 `
        --factor-config $yaml --output-dir $out 2>&1 |
        ForEach-Object {
            $_ | Out-File -FilePath $runLog -Append -Encoding utf8
            $_
        }
    $rc = $LASTEXITCODE
    "DONE $mode h5 rc=$rc $(Get-Date -Format o)" | Tee-Object -Append $driverLog
}
"=== v6 h5 RESUME END $(Get-Date -Format o) ===" | Tee-Object -Append $driverLog
