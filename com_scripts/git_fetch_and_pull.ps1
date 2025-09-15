
git fetch origin

$diffCount = [int](git rev-list HEAD...origin/main --count)
if ($diffCount -gt 0) {
  Write-Host "Updates found, forcing reset to origin/main..."
  git reset --hard origin/main
} else {
  Write-Host "No updates"
}