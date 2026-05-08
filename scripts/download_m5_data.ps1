New-Item -ItemType Directory -Force -Path data | Out-Null

$zipPath = "data\m5-forecasting-accuracy.zip"
$extractPath = "data\m5"
$url = "https://zenodo.org/records/12636070/files/m5-forecasting-accuracy.zip?download=1"

if (-not (Test-Path $zipPath)) {
    Invoke-WebRequest -Uri $url -OutFile $zipPath
}

New-Item -ItemType Directory -Force -Path $extractPath | Out-Null
Expand-Archive -Force -Path $zipPath -DestinationPath $extractPath

Write-Host "M5 data extracted to $extractPath"
