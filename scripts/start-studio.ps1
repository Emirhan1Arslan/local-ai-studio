<#
Local AI Studio - tek tusla ayaga kaldir / durdur. (Dosya bilerek ASCII: Windows
PowerShell 5.1 BOM'suz UTF-8'i ANSI okur, Turkce karakterler parse hatasi verir.)

  scripts\start-studio.ps1            backend (8710) + UI (5173) ayri pencerelerde, tarayici acilir
  scripts\start-studio.ps1 -Stop      ikisini de kapat
  scripts\start-studio.ps1 -NoBrowser tarayici acma
  scripts\start-studio.ps1 -Lan       API'yi 0.0.0.0'a bagla (LAN modu - token API penceresinde yazilir)

Zaten calisan bilesen yeniden baslatilmaz (health/port kontrolu). Masaustu kisayolu
icin: scripts\install-shortcut.ps1
#>
param(
    [switch]$Stop,
    [switch]$NoBrowser,
    [switch]$Lan
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ApiPort = 8710
$UiPort = 5173
$ApiUrl = "http://127.0.0.1:$ApiPort"
$UiUrl = "http://localhost:$UiPort"

function Get-ListenerPid([int]$port) {
    $line = netstat -ano | Select-String -Pattern ":$port\s+.*LISTENING\s+(\d+)" | Select-Object -First 1
    if ($line) { return [int]$line.Matches[0].Groups[1].Value }
    return $null
}

function Test-Url([string]$url) {
    try {
        $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2
        return ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500)
    } catch { return $false }
}

if ($Stop) {
    foreach ($port in @($ApiPort, $UiPort)) {
        $procId = Get-ListenerPid $port
        if ($procId) {
            Write-Host "port $port -> PID $procId kapatiliyor"
            taskkill /F /T /PID $procId | Out-Null
        } else {
            Write-Host "port $port zaten bos"
        }
    }
    Write-Host "Local AI Studio durduruldu."
    exit 0
}

Set-Location $Root
Write-Host "Local AI Studio baslatiliyor ($Root)"

# --- backend -------------------------------------------------------------------
if (Test-Url "$ApiUrl/v1/health") {
    Write-Host "API zaten calisiyor: $ApiUrl"
} else {
    $apiArgs = "uv run studio-api --port $ApiPort"
    if ($Lan) { $apiArgs += " --host 0.0.0.0 --token auto" }
    Start-Process powershell -WorkingDirectory $Root -ArgumentList @(
        "-NoExit", "-Command",
        "`$Host.UI.RawUI.WindowTitle = 'Studio API ($ApiPort)'; $apiArgs"
    ) | Out-Null
    Write-Host "API penceresi acildi ($apiArgs)"
}

# --- UI (Vite dev) -------------------------------------------------------------
if (Get-ListenerPid $UiPort) {
    Write-Host "UI zaten calisiyor: $UiUrl"
} else {
    Start-Process powershell -WorkingDirectory $Root -ArgumentList @(
        "-NoExit", "-Command",
        "`$Host.UI.RawUI.WindowTitle = 'Studio UI ($UiPort)'; pnpm --filter desktop dev"
    ) | Out-Null
    Write-Host "UI penceresi acildi (pnpm --filter desktop dev)"
}

# --- hazir olmayi bekle --------------------------------------------------------
$deadline = (Get-Date).AddSeconds(90)
$apiOk = $false
$uiOk = $false
while ((Get-Date) -lt $deadline) {
    if (-not $apiOk) { $apiOk = Test-Url "$ApiUrl/v1/health" }
    if (-not $uiOk) { $uiOk = Test-Url $UiUrl }
    if ($apiOk -and $uiOk) { break }
    Start-Sleep -Milliseconds 800
}
if ($apiOk) { Write-Host "API : hazir  $ApiUrl  (MCP: $ApiUrl/mcp)" } else { Write-Host "API : HENUZ YOK - API penceresine bakin" }
if ($uiOk) { Write-Host "UI  : hazir  $UiUrl" } else { Write-Host "UI  : HENUZ YOK - UI penceresine bakin" }

if ($uiOk -and -not $NoBrowser) {
    Start-Process $UiUrl
}
