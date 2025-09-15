### Remote auto-pull setup

These commands are meant to be run on the OTHER server. Replace `YOUR_USER` and `YOUR_HOST` if you copy scp/ssh examples.

- **Working tree**: /data1/EAT_projs/auto_repo
- **Bare repo**: /data1/EAT_projs/auto_repo.git

#### 1) Create bare repository that will receive pushes
```bash
sudo mkdir -p /data1/EAT_projs
sudo chown -R "$USER":"$USER" /data1/EAT_projs
cd /data1/EAT_projs
[ -d auto_repo.git ] || git init --bare auto_repo.git
```

#### 2) Create/prepare the working tree (checked-out copy)
```bash
mkdir -p /data1/EAT_projs/auto_repo
cd /data1/EAT_projs/auto_repo

# If this directory is empty, you can do a first clone from the bare repo:
# (run after step 3 hook is installed, or just clone now and fetch later)
if [ -z "$(ls -A)" ]; then
  git clone /data1/EAT_projs/auto_repo.git .
fi
```

#### 3) Install server-side post-receive hook to auto-update working tree
Create the hook file:
```bash
cat > /data1/EAT_projs/auto_repo.git/hooks/post-receive <<'SH'
#!/usr/bin/env bash
set -euo pipefail

WORK_TREE="/data1/EAT_projs/auto_repo"
GIT_DIR="/data1/EAT_projs/auto_repo.git"

export GIT_DIR

# Ensure working tree exists
mkdir -p "$WORK_TREE"

# Update working tree to the latest pushed commit on main (or current default)
# Determine default branch from the push; fall back to main if unknown.
read -r oldrev newrev refname || true
branch="${refname##refs/heads/}"
if [ -z "${branch:-}" ]; then
  branch="main"
fi

# Checkout/update
/usr/bin/env git --work-tree="$WORK_TREE" checkout -f "$branch"
/usr/bin/env git --work-tree="$WORK_TREE" reset --hard "$newrev"
/usr/bin/env git --work-tree="$WORK_TREE" submodule sync --recursive || true
/usr/bin/env git --work-tree="$WORK_TREE" submodule update --init --recursive || true

# Optional: permissions fixups
# chown -R www-data:www-data "$WORK_TREE"
# find "$WORK_TREE" -type d -exec chmod 755 {} +
# find "$WORK_TREE" -type f -exec chmod 644 {} +
SH
chmod +x /data1/EAT_projs/auto_repo.git/hooks/post-receive
```

Notes:
- The hook reads the branch that was pushed and updates the working tree accordingly.
- If you only want to auto-deploy `main`, replace the checkout lines with:
```bash
[ "$refname" = "refs/heads/main" ] || exit 0
/usr/bin/env git --work-tree="$WORK_TREE" checkout -f main
/usr/bin/env git --work-tree="$WORK_TREE" reset --hard "$newrev"
```

#### 4) From this machine, add the remote and push
On your local machine (this one):
```bash
cd /Users/kuhn/Documents/code/auto_repo
# Add remote pointing to the remote bare repo (adjust ssh if needed):
# Using direct path over SSH:
# git remote add prod "ssh://YOUR_USER@YOUR_HOST/data1/EAT_projs/auto_repo.git"
# Or if using a shared filesystem / VPN, you can mount and use a file path.

# Set upstream and push main
# git push -u prod main
```

#### 5) Verify
- Make a commit locally; it should auto-push (post-commit hook).
- The remote bare repo receives the push, triggers the `post-receive` hook, and updates `/data1/EAT_projs/auto_repo`.

Troubleshooting:
- Ensure the user that runs the `git-receive-pack` on the remote has write perms on `/data1/EAT_projs/auto_repo.git` and `/data1/EAT_projs/auto_repo`.
- Check hook logs by temporarily adding `set -x` at the top of the hook.
- Hooks run non-interactively: set up SSH keys and non-interactive auth.
