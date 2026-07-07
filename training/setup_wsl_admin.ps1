<#
  Step 1 of 2 — RUN THIS AS ADMINISTRATOR (right-click > Run as administrator,
  or from an elevated PowerShell:  powershell -ExecutionPolicy Bypass -File setup_wsl_admin.ps1)

  Installs WSL2 + Ubuntu. This is the ONLY step that needs admin rights and a
  reboot. After it finishes and you reboot, Ubuntu will do its first-run setup
  (it asks you to pick a UNIX username + password — any values are fine), then
  run step 2:  bash /mnt/c/Users/arvin/Arvind-Lora/training/wsl_provision.sh
#>

Write-Host "== Installing WSL2 + Ubuntu ==" -ForegroundColor Cyan

# --no-launch installs the distro without immediately opening it, so the reboot
# happens cleanly first. On older builds that don't support the flag, fall back.
wsl --install -d Ubuntu --no-launch
if ($LASTEXITCODE -ne 0) {
    Write-Host "Retrying without --no-launch ..." -ForegroundColor Yellow
    wsl --install
}

Write-Host ""
Write-Host "== Enabling required Windows features (idempotent) ==" -ForegroundColor Cyan
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart | Out-Null
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart | Out-Null

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host " DONE. Now REBOOT Windows." -ForegroundColor Green
Write-Host " After reboot:" -ForegroundColor Green
Write-Host "   1. Ubuntu opens and asks for a username + password (pick anything)." -ForegroundColor Green
Write-Host "   2. In that Ubuntu window, paste:" -ForegroundColor Green
Write-Host "        bash /mnt/c/Users/arvin/Arvind-Lora/training/wsl_provision.sh" -ForegroundColor White
Write-Host "=====================================================" -ForegroundColor Green
