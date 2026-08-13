[CmdletBinding()]
param(
    [ValidateSet("cpu", "gpu")]
    [string]$Profile = "cpu"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Test-Path ".env.local")) {
    throw "Fichier .env.local absent : aucune installation locale LISA identifiable."
}

$composeArgs = @("compose", "--env-file", ".env.local", "-f", "compose.test.yaml")
if ($Profile -eq "gpu") {
    $composeArgs += @("-f", "compose.gpu.yaml")
}
$composeArgs += "down"
& docker @composeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "LISA est arrêtée. Les données, modèles et volumes sont conservés."
