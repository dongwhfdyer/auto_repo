# Auto-pull setup on the other server

This server (developer box) auto-pushes to GitHub after each commit. The other server should keep `/data1/EAT_projs/auto_repo` up to date by pulling from GitHub. Below are two options.

Before you start, ensure the other server can authenticate to GitHub (recommended: SSH deploy key; alternative: HTTPS with a PAT).

## Option A: Cron-based auto-pull (simple and robust)

1) Create the working directory if missing and clone once:

```bash
sudo mkdir -p /data1/EAT_projs && sudo chown -R "$USER":"$USER" /data1/EAT_projs
cd /data1/EAT_projs
# Replace REPO_URL with your GitHub URL (SSH example shown)
# e.g. git@github.com:YOURORG/auto_repo.git
REPO_URL="git@github.com:YOURORG/auto_repo.git"
if [ ! -d auto_repo/.git ]; then
  git clone "$REPO_URL" auto_repo
fi
```

2) Add a cron job to pull every minute:

```bash
crontab -l 2>/dev/null | { cat; echo "* * * * * cd /data1/EAT_projs/auto_repo && git fetch --all --prune && git reset --hard origin/main >> /data1/EAT_projs/auto_repo/.git/cron-pull.log 2>&1"; } | crontab -
```

- This does a fast, safe sync to `origin/main`. Adjust the branch if needed.
- Log file: `/data1/EAT_projs/auto_repo/.git/cron-pull.log`.

## Option B: Hook-based auto-pull (pull on demand)

If you prefer only updating when someone runs `git pull` or after merges on that server, add a `post-merge` hook to auto-reset to origin:

```bash
cat > /data1/EAT_projs/auto_repo/.git/hooks/post-merge <<'SH'
#!/usr/bin/env bash
set -euo pipefail
branch=$(git rev-parse --abbrev-ref HEAD)
# Ensure we are aligned with origin after merges
if git ls-remote --exit-code origin "refs/heads/${branch}" >/dev/null 2>&1; then
  git fetch --all --prune
  git reset --hard "origin/${branch}"
fi
SH
chmod +x /data1/EAT_projs/auto_repo/.git/hooks/post-merge
```

## Authentication options

- SSH (recommended): Add the server's public key to GitHub Deploy Keys or use a machine user. Test with:

```bash
ssh -T git@github.com
```

- HTTPS with PAT: Set the remote to `https://github.com/YOURORG/auto_repo.git` and use a credential helper:

```bash
git config --global credential.helper store
# Next pull/push will prompt once and store the token
```

## Keeping local changes safe

The cron job uses `git reset --hard origin/main`, which overwrites local changes. If you need local modifications on the other server, either:

- Keep changes on a separate branch and rebase/merge as needed, or
- Replace the cron pull line with a safer fast-forward attempt:

```bash
cd /data1/EAT_projs/auto_repo && git pull --ff-only
```

## Set remote origin (if missing)

If the remote wasn't recorded during clone (or to switch to SSH):

```bash
cd /data1/EAT_projs/auto_repo
REPO_URL="git@github.com:YOURORG/auto_repo.git"
if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi
```

