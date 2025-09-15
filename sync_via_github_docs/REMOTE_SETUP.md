# Windows remote auto-pull setup

Target working copy: `C:\User\xiajiakun\Documents\code\auto_repo`

This guide uses GitHub as the bridge. It keeps the working copy synced to `origin/main` automatically using a Windows Scheduled Task.

## Prerequisites

- Install Git for Windows (`git` available in PATH).
- The Windows machine can authenticate to GitHub (SSH key or HTTPS with PAT via credential manager).
- PowerShell 5+.

## 1) Initial clone (one-time)

Open PowerShell as the user that will run the scheduled task, then:

```powershell
$repoUrl = "git@github.com:YOURORG/auto_repo.git"  # or HTTPS URL
$root = "C:\\User\\xiajiakun\\Documents\\code"
$work = Join-Path $root "auto_repo"

if (-not (Test-Path $root)) { New-Item -ItemType Directory -Force -Path $root | Out-Null }
if (-not (Test-Path (Join-Path $work ".git"))) {
  git clone $repoUrl $work
}
```

If you need to set/change the remote later:

```powershell
$repoUrl = "git@github.com:YOURORG/auto_repo.git"
$work = "C:\\User\\xiajiakun\\Documents\\code\\auto_repo"
if (git -C $work remote get-url origin 2>$null) {
  git -C $work remote set-url origin $repoUrl
} else {
  git -C $work remote add origin $repoUrl
}
```

## 2) Create a sync script

Save this PowerShell script at `C:\User\xiajiakun\Documents\code\auto_repo\sync.ps1`:

```powershell
Param(
  [string]$Branch = "main"
)
$ErrorActionPreference = "Stop"
$work = "C:\\User\\xiajiakun\\Documents\\code\\auto_repo"
$log  = Join-Path $work ".git\\sync.log"

# Ensure directory exists
if (-not (Test-Path $work)) { throw "Working directory not found: $work" }

# Do a safe fast sync to origin/$Branch
& git -C $work fetch --all --prune *>> $log
& git -C $work reset --hard "origin/$Branch" *>> $log
```

Notes:
- This will overwrite local changes. If you must preserve local edits, replace `reset --hard` with `pull --ff-only`.

## 3) Create a Scheduled Task to run every minute

Run the following once in an elevated PowerShell prompt (Administrator) to register a per-user task:

```powershell
$taskName = "AutoRepoSync"
$script   = "C:\\User\\xiajiakun\\Documents\\code\\auto_repo\\sync.ps1"
$action   = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`" -Branch main"
$trigger  = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 1) -RepetitionDuration ([TimeSpan]::MaxValue)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Description "Pull origin/main into auto_repo every minute" -User $env:USERNAME -RunLevel Limited -Force
```

- The task runs invisibly every minute. Logs are in `C:\User\xiajiakun\Documents\code\auto_repo\.git\sync.log`.
- If you prefer to run as SYSTEM or a service account, adjust `-User` accordingly.

## 4) Test manually

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File "C:\\User\\xiajiakun\\Documents\\code\\auto_repo\\sync.ps1" -Branch main
Get-Content -Tail 50 "C:\\User\\xiajiakun\\Documents\\code\\auto_repo\\.git\\sync.log"
```

## Optional: SSH authentication setup

Generate an SSH key and add to GitHub (Deploy Key or your account):

```powershell
ssh-keygen -t ed25519 -C "windows-remote-auto-repo"
# Start agent and add key (Git for Windows ships ssh-agent service)
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent
ssh-add $HOME\.ssh\id_ed25519
ssh -T git@github.com
```

If using HTTPS + PAT, Git Credential Manager will prompt on first pull and cache your credentials.

