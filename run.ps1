$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    & (Join-Path $projectRoot "install.ps1") -Launch
    exit $LASTEXITCODE
}
& $venvPython (Join-Path $projectRoot "main.py")
exit $LASTEXITCODE
