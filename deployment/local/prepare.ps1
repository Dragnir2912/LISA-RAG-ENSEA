[CmdletBinding()]
param(
    [ValidateSet("cpu", "gpu")]
    [string]$Profile = "cpu"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $projectRoot

function Assert-LastCommand([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step a échoué (code $LASTEXITCODE). Consultez le message Docker affiché juste au-dessus."
    }
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker est introuvable. Installez et démarrez Docker Desktop."
}
docker version | Out-Null
Assert-LastCommand "La vérification de Docker"
docker compose version | Out-Null
Assert-LastCommand "La vérification de Docker Compose"

if (-not (Test-Path ".env.local")) {
    Copy-Item ".env.local.example" ".env.local"
    Write-Host "Fichier .env.local créé depuis l'exemple."
}

New-Item -ItemType Directory -Force `
    "secrets", "models\huggingface", "models\ollama" | Out-Null

if (-not (Test-Path "secrets\postgres_password.txt")) {
    $passwordBytes = New-Object byte[] 48
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($passwordBytes) } finally { $rng.Dispose() }
    [Convert]::ToBase64String($passwordBytes) |
        Set-Content -NoNewline "secrets\postgres_password.txt"
}
if (-not (Test-Path "secrets\application_key.txt")) {
    $keyBytes = New-Object byte[] 32
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($keyBytes) } finally { $rng.Dispose() }
    $fernetKey = [Convert]::ToBase64String($keyBytes).Replace('+','-').Replace('/','_')
    $fernetKey | Set-Content -NoNewline "secrets\application_key.txt"
}

docker compose --env-file .env.local -f compose.test.yaml build api frontend
Assert-LastCommand "La construction des images LISA"
docker compose --env-file .env.local -f compose.test.yaml pull postgres ollama
Assert-LastCommand "Le téléchargement des images PostgreSQL et Ollama"

$modelCache = (Resolve-Path "models\huggingface").Path
$imageTag = "lisa-api:local"
$imageTagLine = Get-Content ".env.local" | Where-Object { $_ -match '^LISA_IMAGE_TAG=' } | Select-Object -First 1
if ($imageTagLine) {
    $configuredTag = ($imageTagLine -split '=', 2)[1].Trim()
    if ($configuredTag) { $imageTag = $configuredTag }
}
docker run --rm `
    -e HF_HOME=/srv/lisa/models/huggingface `
    -e HF_HUB_OFFLINE=0 `
    -e TRANSFORMERS_OFFLINE=0 `
    --mount "type=bind,source=$modelCache,target=/srv/lisa/models/huggingface" `
    $imageTag python /srv/lisa/deployment/prefetch_models.py
Assert-LastCommand "Le préchargement des modèles d'embedding et de reranking"

docker compose --env-file .env.local -f compose.test.yaml up -d ollama
Assert-LastCommand "Le démarrage temporaire d'Ollama"
try {
    docker compose --env-file .env.local -f compose.test.yaml exec -T ollama `
        ollama pull qwen3.5:9b
    Assert-LastCommand "Le téléchargement de qwen3.5:9b"
    docker compose --env-file .env.local -f compose.test.yaml exec -T ollama `
        ollama show qwen3.5:9b | Out-Null
    Assert-LastCommand "La vérification de qwen3.5:9b"
}
finally {
    docker compose --env-file .env.local -f compose.test.yaml stop ollama
}

Write-Host "Préparation terminée pour le profil $Profile."
Write-Host "Lancez : .\deployment\local\start.ps1 -Profile $Profile"
