# In the repo directory (or add: git -C "C:\path\to\repo" ...)
git fetch origin

$diffCount = [int](git rev-list HEAD...origin/main --count)
if ($diffCount -gt 0) {
  Write-Host "Updates found, pulling..."
  git pull
} else {
  Write-Host "No updates"
}