# build.ps1
$ErrorActionPreference = "Stop"

$initPath = "lightrag/__init__.py"
if (!(Test-Path $initPath)) {
  Write-Error "Fichier introuvable: $initPath"
}

# Récupère __version__ = "x.y.z"
$regex = '__version__\s*=\s*["'']([^"'']+)'
$match = Select-String -Path $initPath -Pattern $regex -AllMatches | Select-Object -First 1
if (-not $match) { Write-Error "Impossible de trouver __version__ dans $initPath" }

$version = $match.Matches[0].Groups[1].Value
if ([string]::IsNullOrWhiteSpace($version)) { Write-Error "Version vide" }

# Nom d'image local (modifiable)
$imageName = "lightrag"
$tagged = "${imageName}:${version}"   # <-- corrigé

Write-Host ">> Version détectée: $version"
Write-Host ">> Construction de l'image: $tagged"

podman build `
  --file Dockerfile `
  --tag $tagged `
  --build-arg LIGHTRAG_VERSION=$version `
  .

# Tag optionnel 'latest'
podman tag $tagged "${imageName}:latest"

# Optionnel: préparer un .env pour compose
"LIGHTRAG_VERSION=$version" | Set-Content -Encoding ascii .env

Write-Host "OK. Images locales:"
podman images --format "table {{.Repository}}\t{{.Tag}}\t{{.Created}}"
