
while ($true) {
  # If using $repo: git -C $repo fetch origin
  git fetch origin

  $diffCount = [int](git rev-list HEAD...origin/main --count)
  if ($diffCount -gt 0) {
    Write-Host "Updates found, forcing reset to origin/main..."
    # If using $repo: git -C $repo reset --hard origin/main
    git reset --hard origin/main
  } else {
    Write-Host "No updates"
  }

  Start-Sleep -Seconds 10
}