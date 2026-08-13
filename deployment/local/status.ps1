[CmdletBinding()]
param(
    [switch]$Follow
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

if (-not (Test-Path ".env.local")) {
    throw "Fichier .env.local absent. Exécutez d'abord .\deployment\local\prepare.ps1."
}

docker compose --env-file .env.local -f compose.test.yaml ps
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$logArgs = @(
    "compose", "--env-file", ".env.local", "-f", "compose.test.yaml",
    "logs", "--tail=80"
)
if ($Follow) { $logArgs += "--follow" }
$logArgs += @("api", "worker", "ollama", "postgres")
& docker @logArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
