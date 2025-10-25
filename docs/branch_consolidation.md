# Consolidating Branches into `main`

Use this checklist to collapse the work from every feature branch into a single, up-to-date `main` branch.

## 1. Update local information

```bash
# Fetch the latest refs from origin (optional if branches are local-only)
git fetch --all --prune

# Inspect local and remote branches
git branch            # local branches
git branch -r         # remote branches
```

If any branches exist only on the remote, create local copies so you can merge them (replace `<branch>` with each branch name):

```bash
git checkout -b <branch> origin/<branch>
```

## 2. Choose the target branch

Decide which branch should become the single source of truth—usually `main`. Checkout and update it before merging:

```bash
git checkout main
git pull              # pulls from origin/main if a remote exists
```

If your local branch is named something else (for example `work`), rename it before you begin merging:

```bash
git branch -m work main          # rename current branch to main
# If a remote already has a main branch you want to replace, use:
# git push origin :main && git push origin main --set-upstream
```

## 3. Merge each branch into `main`

For each branch you want to preserve, merge it into `main` one at a time. Resolve conflicts as they arise.

```bash
git checkout main

for BR in \
  codex/change-github-led-to-dimmest-red \
  codex/fix-font-size-in-Draw_inside.py \
  codex/fix-font-size-inside \
  codex/investigate-desk_display-service-crash \
  codex/fix-font-loading-error-to-notcoloremoji.ttf \
  codex/create-installer-script-for-project
  do
    git merge "$BR"
    # resolve conflicts if needed, then run:
    # git add <resolved-files>
    # git commit
  done
```

*Tip:* If you want a linear history, replace `git merge "$BR"` with `git merge --squash "$BR"` and follow it with `git commit` to author a single commit per branch.

## 4. Verify and clean up

After all merges succeed:

```bash
# Confirm every branch is merged
git log --graph --oneline --decorate

git status           # ensure no remaining staged changes
```

When you are satisfied that `main` contains all the desired work, delete the feature branches both locally and on the remote to declutter:

```bash
# delete local branches safely (use -D instead of -d if Git warns about unmerged changes)
for BR in \
  codex/change-github-led-to-dimmest-red \
  codex/fix-font-size-in-Draw_inside.py \
  codex/fix-font-size-inside \
  codex/investigate-desk_display-service-crash \
  codex/fix-font-loading-error-to-notcoloremoji.ttf \
  codex/create-installer-script-for-project
  do
    git branch -d "$BR"
  done

# delete remote branches
for BR in \
  codex/change-github-led-to-dimmest-red \
  codex/fix-font-size-in-Draw_inside.py \
  codex/fix-font-size-inside \
  codex/investigate-desk_display-service-crash \
  codex/fix-font-loading-error-to-notcoloremoji.ttf \
  codex/create-installer-script-for-project
  do
    git push origin --delete "$BR"
  done
```

Finally, push the consolidated `main` branch:

```bash
git push origin main
```

## 5. Optional: archive before deleting

If you are unsure about deleting branches, create tags pointing at their tips so you can restore them later:

```bash
for BR in \
  codex/change-github-led-to-dimmest-red \
  codex/fix-font-size-in-Draw_inside.py \
  codex/fix-font-size-inside \
  codex/investigate-desk_display-service-crash \
  codex/fix-font-loading-error-to-notcoloremoji.ttf \
  codex/create-installer-script-for-project
  do
    git tag "archive/${BR##*/}" "$BR"
  done

git push origin --tags
```

These tags act as read-only snapshots that you can revisit even after deleting the branches.
