git fetch origin

# Check if there are new commits
if [ $(git rev-list HEAD...origin/main --count) -gt 0 ]; then
    echo "Updates found, pulling..."
    git pull
else
    echo "No updates"
fi