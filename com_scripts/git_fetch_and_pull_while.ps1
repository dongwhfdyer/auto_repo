# Optional: set working repo path
# $repo = "C:\User\xiajiakun\Documents\code\auto_repo"

while ($true) {
  # If using $repo: git -C $repo fetch origin
  git fetch origin

  $diffCount = [int](git rev-list HEAD...origin/main --count)
  if ($diffCount -gt 0) {
    Write-Host "Updates found, pulling..."
    # If using $repo: git -C $repo pull
    git pull
  } else {
    Write-Host "No updates"
  }

  Start-Sleep -Seconds 10
}