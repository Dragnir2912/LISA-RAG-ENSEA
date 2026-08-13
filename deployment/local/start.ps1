[CmdletBinding()]
param(
    [ValidateSet("cpu", "gpu")]
    [string]$Profile = "cpu",
    [switch]$Build
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker est introuvable. Installez et démarrez Docker Desktop."
}
docker version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker Desktop ne répond pas." }
docker compose version | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Le module Docker Compose est indisponible." }

if (-not (Test-Path ".env.local")) {
    throw "Fichier .env.local absent. Exécutez d'abord deployment\local\prepare.ps1."
}
foreach ($required in @(
    "secrets\postgres_password.txt",
    "secrets\application_key.txt",
    "models\huggingface",
    "models\ollama",
    "data\embeddings"
)) {
    if (-not (Test-Path $required)) {
        throw "Élément requis absent : $required. Reprenez la préparation documentée."
    }
}

$composeArgs = @("compose", "--env-file", ".env.local", "-f", "compose.test.yaml")
if ($Profile -eq "gpu") {
    $composeArgs += @("-f", "compose.gpu.yaml")
}
$composeArgs += @("up", "-d")
$composeArgs += if ($Build) { "--build" } else { "--no-build" }

& docker @composeArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& docker compose --env-file .env.local -f compose.test.yaml ps
Write-Host "LISA est disponible sur http://localhost:8080"
