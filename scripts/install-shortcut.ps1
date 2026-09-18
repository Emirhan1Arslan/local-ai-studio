<#
Masaustune "Local AI Studio" kisayolu koyar (start-studio.bat'i hedefler).
Bir kez calistirilir; kisayol Tauri ikonunu kullanir.
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Desktop = [Environment]::GetFolderPath("Desktop")
$LinkPath = Join-Path $Desktop "Local AI Studio.lnk"
$Target = Join-Path $Root "start-studio.bat"
$Icon = Join-Path $Root "apps\desktop\src-tauri\icons\icon.ico"

$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut($LinkPath)
$lnk.TargetPath = $Target
$lnk.WorkingDirectory = $Root
$lnk.Description = "Local AI Studio: backend + UI'yi baslat"
if (Test-Path $Icon) { $lnk.IconLocation = $Icon }
$lnk.Save()
Write-Host "Kisayol olusturuldu: $LinkPath"
