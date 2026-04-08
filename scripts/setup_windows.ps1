param(
    [string]$VenvName = "soft_comp"
)

$ErrorActionPreference = "Stop"

Write-Host "Creating virtual environment '$VenvName' on Windows..."
python -m venv $VenvName

$pythonExe = Join-Path $VenvName "Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment python executable not found at $pythonExe"
}

Write-Host "Upgrading pip..."
& $pythonExe -m pip install --upgrade pip

Write-Host "Installing project dependencies..."
& $pythonExe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate with:"
Write-Host ".\$VenvName\Scripts\activate"
