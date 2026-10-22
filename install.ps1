param(
    [switch]$CpuOnly,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

function Find-Python {
    foreach ($name in @("py", "python", "python3")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command) {
            if ($name -eq "py") {
                return @{ Executable = $command.Source; Arguments = @("-3") }
            }
            return @{ Executable = $command.Source; Arguments = @() }
        }
    }
    $storeShim = Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe"
    if (Test-Path $storeShim) {
        return @{ Executable = $storeShim; Arguments = @() }
    }
    throw "Python 3.10-3.12 is required. Install Python from python.org and enable its PATH option."
}

if (-not (Test-Path $venvPython)) {
    $pythonCommand = Find-Python
    $basePython = $pythonCommand.Executable
    $pythonArgs = $pythonCommand.Arguments
    $version = & $basePython @pythonArgs -c 'import sys; print(str(sys.version_info.major)+chr(46)+str(sys.version_info.minor))'
    if ($LASTEXITCODE -ne 0) { throw "Could not run Python: $basePython" }
    if ($version -notmatch '^3\.(10|11|12)$') {
        throw "Detected Python $version. Use Python 3.10, 3.11, or 3.12."
    }
    & $basePython @pythonArgs -m venv (Join-Path $projectRoot ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the virtual environment." }
}

Push-Location $projectRoot
try {
    & $venvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install the desktop dependencies." }

    $gpuCommand = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $CpuOnly -and $gpuCommand) {
        Write-Host "Installing PyTorch CUDA 12.1 wheels for NVIDIA inference."
        & $venvPython -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    }
    else {
        Write-Host "Installing CPU-only PyTorch wheels."
        & $venvPython -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
    }
    if ($LASTEXITCODE -ne 0) { throw "Could not install the selected PyTorch build." }
    & $venvPython -m pip install -r requirements-ai.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install the AI processing dependencies." }

    & $venvPython -c 'import torch, PySide6, PIL; print(torch.__version__); print(torch.cuda.is_available())'
    if ($LASTEXITCODE -ne 0) { throw "The installation verification failed." }
    New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot "input"), (Join-Path $projectRoot "output") | Out-Null
    if ($Launch) {
        & $venvPython (Join-Path $projectRoot "main.py")
        if ($LASTEXITCODE -ne 0) { throw "The application exited with an error." }
    }
    else {
        Write-Host "Setup complete. Start the app with .\run.ps1"
    }
}
finally {
    Pop-Location
}
