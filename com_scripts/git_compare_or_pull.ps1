Param(
  [string]$RepoPath = "C:\Users\xiajiakun\Documents\code\auto_repo",
  [string]$Branch   = "main",
  [switch]$CompareOnly,     # only show differences, do not modify
  [switch]$Force            # overwrite local changes with origin/Branch
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $RepoPath -PathType Container)) {
  $fallback = Join-Path $env:USERPROFILE "Documents\code\auto_repo"
  if (Test-Path -Path $fallback -PathType Container) { $RepoPath = $fallback }
  else { throw "Repo path not found. Tried: '$RepoPath' and '$fallback'" }
}

Set-Location -Path $RepoPath

# Ensure we can reach origin and know the target branch
git fetch origin --prune | Out-Null

# How many commits differ between local HEAD and origin/Branch
$diffCount = [int](git rev-list HEAD...("origin/" + $Branch) --count)
Write-Host ("Diff commits vs origin/{0}: {1}" -f $Branch, $diffCount)

if ($CompareOnly) {
  if ($diffCount -gt 0) {
    Write-Host "Showing top 20 differing commits (local vs origin):"
    git log --oneline --decorate --graph -n 20 HEAD..("origin/" + $Branch)
  }
  exit 0
}

if ($diffCount -le 0) {
  Write-Host "No updates"
  exit 0
}

if ($Force) {
  Write-Host ("Updates found, forcing reset to origin/{0}..." -f $Branch)
  git reset --hard ("origin/" + $Branch)
} else {
  Write-Host "Updates found, pulling (fast-forward if possible)..."
  git pull --ff-only || (
    Write-Host "Fast-forward not possible. Re-run with -Force to overwrite local changes."; exit 1
  )
}


