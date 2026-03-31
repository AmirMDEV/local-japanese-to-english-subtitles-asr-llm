param(
    [string]$VenvPath = ".venv311",
    [string]$DistName = "SubtitleTool"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$venvPython = Join-Path $repoRoot "$VenvPath\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python was not found at $venvPython. Run scripts\bootstrap.ps1 first."
}

Write-Host "Installing build dependencies..."
& $venvPython -m pip install pyinstaller

$buildDir = Join-Path $repoRoot "build"
$distDir = Join-Path $repoRoot "dist"
$releaseRoot = Join-Path $distDir $DistName
if (Test-Path $buildDir) {
    Remove-Item -LiteralPath $buildDir -Recurse -Force
}
if (Test-Path $releaseRoot) {
    Remove-Item -LiteralPath $releaseRoot -Recurse -Force
}
Get-ChildItem -LiteralPath $distDir -Filter "$DistName-windows-x64-*.zip" -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force
}

Write-Host "Building Windows app bundle..."
& $venvPython -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name $DistName `
    --paths src `
    --collect-submodules local_subtitle_stack `
    --hidden-import torch `
    --hidden-import transformers `
    --hidden-import tokenizers `
    --hidden-import accelerate `
    --hidden-import safetensors `
    --exclude-module pytest `
    --exclude-module torchvision `
    --exclude-module torchaudio `
    --exclude-module tensorboard `
    src\local_subtitle_stack\launcher.py

if (-not (Test-Path $releaseRoot)) {
    throw "Expected release folder was not created: $releaseRoot"
}

Copy-Item -LiteralPath README.md -Destination (Join-Path $releaseRoot "README.md") -Force

$generatedSpec = Join-Path $repoRoot "$DistName.spec"
if (Test-Path $generatedSpec) {
    Remove-Item -LiteralPath $generatedSpec -Force
}

Write-Host "Creating release zip parts..."
& $venvPython (Join-Path $repoRoot "scripts\package_release_assets.py") --dist-root $releaseRoot --dist-name $DistName

Write-Host ""
Write-Host "Build complete."
Write-Host "App folder: $releaseRoot"
Write-Host "Release assets:"
Get-ChildItem -LiteralPath $distDir -Filter "$DistName-windows-x64-*.zip" | ForEach-Object {
    Write-Host "  $($_.FullName)"
}
