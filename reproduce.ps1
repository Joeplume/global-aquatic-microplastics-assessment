param([string]$Stage="results", [string]$Output="outputs")
$ErrorActionPreference="Stop"
& python (Join-Path $PSScriptRoot "run_reproduce.py") --stage $Stage --output $Output
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
