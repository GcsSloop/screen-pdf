$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$TargetDir = Join-Path $RootDir "program\engine\vendor\windows\bin"
New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

if (-not $env:SCREEN_PDF_RUNTIME_SOURCE_DIR) {
    throw "SCREEN_PDF_RUNTIME_SOURCE_DIR is required for Windows runtime preparation"
}

$SourceDir = Join-Path $env:SCREEN_PDF_RUNTIME_SOURCE_DIR "windows\bin"
$Required = @("python.exe", "tesseract.exe", "gswin64c.exe")
foreach ($Name in $Required) {
    $SourcePath = Join-Path $SourceDir $Name
    if (-not (Test-Path $SourcePath)) {
        throw "missing required runtime input: $SourcePath"
    }
    Copy-Item -Recurse -Force $SourcePath (Join-Path $TargetDir $Name)
}

Write-Host "Prepared Windows runtime under $TargetDir"
