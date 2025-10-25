#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<USAGE
Usage: ${0##*/} [options] [branch ...]

Automate consolidation of feature branches into a target branch (default: main).

Options:
  -t, --target <branch>   Target branch to receive merges (default: main)
  -r, --remote <name>     Remote to sync with (default: origin)
  -s, --squash            Use squash merges instead of regular merges
  -a, --archive           Create archive/<branch> tags at each branch tip before merging
  -c, --cleanup           Delete merged branches locally and on the remote
      --force-cleanup     Use git branch -D when deleting local branches
  -p, --push              Push the updated target branch (and tags if created)
  -h, --help              Show this help message

Positional arguments (optional):
  branch                  Branches to merge. If omitted, all remote branches
                          except the target branch are merged.

Examples:
  ${0##*/}
  ${0##*/} -t release --squash
  ${0##*/} codex/fix-bug codex/new-feature
USAGE
}

abort() {
  echo "Error: $1" >&2
  exit 1
}

ensure_clean_worktree() {
  if [[ -n "$(git status --porcelain)" ]]; then
    abort "Working tree is not clean. Commit, stash, or discard changes before running this script."
  fi
}

ensure_git_repo() {
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || abort "This script must be run from within a git repository."
}

resolve_target_branch() {
  local target=$1
  local remote=$2

  if ! git show-ref --verify --quiet "refs/heads/$target"; then
    if git show-ref --verify --quiet "refs/remotes/$remote/$target"; then
      echo "Creating local $target branch from $remote/$target"
      git checkout -b "$target" "$remote/$target"
    else
      abort "Target branch '$target' does not exist locally or on remote '$remote'."
    fi
  else
    git checkout "$target"
    if git show-ref --verify --quiet "refs/remotes/$remote/$target"; then
      git pull "$remote" "$target"
    fi
  fi
}

collect_branches() {
  local target=$1
  local remote=$2
  shift 2
  local -n _result=$1
  shift
  if [[ $# -gt 0 ]]; then
    _result=("$@")
  else
    mapfile -t _result < <(git for-each-ref --format='%(refname:short)' "refs/remotes/$remote" \
      | grep -v "^$remote/$target$" \
      | grep -v "^$remote/HEAD$")
    for i in "${!_result[@]}"; do
      _result[$i]="${_result[$i]#$remote/}"
    done
  fi
}

ensure_local_branch() {
  local branch=$1
  local remote=$2
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    return 0
  fi
  if git show-ref --verify --quiet "refs/remotes/$remote/$branch"; then
    echo "Creating local branch $branch from $remote/$branch"
    git branch --track "$branch" "$remote/$branch"
  else
    abort "Branch '$branch' not found on remote '$remote'."
  fi
}

merge_branch() {
  local branch=$1
  local target=$2
  local squash=$3
  if [[ "$squash" == "1" ]]; then
    echo "Squash merging $branch into $target"
    git merge --squash "$branch"
    git commit -m "Squash merge branch '$branch' into $target"
  else
    echo "Merging $branch into $target"
    git merge "$branch"
  fi
}

create_archive_tag() {
  local branch=$1
  local commit=$2
  local tag="archive/${branch##*/}"
  if git show-ref --verify --quiet "refs/tags/$tag"; then
    echo "Updating existing tag $tag"
    git tag -f "$tag" "$commit"
  else
    echo "Creating tag $tag"
    git tag "$tag" "$commit"
  fi
}

cleanup_branch() {
  local branch=$1
  local remote=$2
  local force_flag=$3
  local delete_flag="-d"
  if [[ "$force_flag" == "1" ]]; then
    delete_flag="-D"
  fi
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    git branch "$delete_flag" "$branch" || echo "Warning: failed to delete local branch $branch" >&2
  fi
  if git ls-remote --exit-code "$remote" "$branch" >/dev/null 2>&1; then
    git push "$remote" --delete "$branch" || echo "Warning: failed to delete remote branch $branch" >&2
  fi
}

main() {
  local target_branch="main"
  local remote="origin"
  local squash=0
  local archive=0
  local cleanup=0
  local force_cleanup=0
  local push_changes=0
  local branches=()
  local positional=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -t|--target)
        target_branch="$2"
        shift 2
        ;;
      -r|--remote)
        remote="$2"
        shift 2
        ;;
      -s|--squash)
        squash=1
        shift
        ;;
      -a|--archive)
        archive=1
        shift
        ;;
      -c|--cleanup)
        cleanup=1
        shift
        ;;
      --force-cleanup)
        force_cleanup=1
        shift
        ;;
      -p|--push)
        push_changes=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        positional+=("$@")
        break
        ;;
      -*)
        abort "Unknown option: $1"
        ;;
      *)
        positional+=("$1")
        shift
        ;;
    esac
  done

  ensure_git_repo
  ensure_clean_worktree

  echo "Fetching latest refs from $remote"
  git fetch --all --prune

  resolve_target_branch "$target_branch" "$remote"

  local -a branches_to_merge
  collect_branches "$target_branch" "$remote" branches_to_merge "${positional[@]}"

  if [[ ${#branches_to_merge[@]} -eq 0 ]]; then
    echo "No branches to merge. Exiting."
    exit 0
  fi

  declare -A branch_tips=()

  for branch in "${branches_to_merge[@]}"; do
    ensure_local_branch "$branch" "$remote"
    branch_tips["$branch"]="$(git rev-parse "$branch")"
  done

  git checkout "$target_branch"

  trap 'echo "\nA command failed. Resolve any issues (for example merge conflicts) and re-run the script." >&2' ERR

  for branch in "${branches_to_merge[@]}"; do
    merge_branch "$branch" "$target_branch" "$squash"
  done

  trap - ERR

  if [[ "$archive" == "1" ]]; then
    for branch in "${branches_to_merge[@]}"; do
      create_archive_tag "$branch" "${branch_tips[$branch]}"
    done
  fi

  if [[ "$cleanup" == "1" ]]; then
    for branch in "${branches_to_merge[@]}"; do
      cleanup_branch "$branch" "$remote" "$force_cleanup"
    done
  fi

  if [[ "$push_changes" == "1" ]]; then
    git push "$remote" "$target_branch"
    if [[ "$archive" == "1" ]]; then
      git push "$remote" --tags
    fi
  fi

  echo "Branch consolidation complete."
}

main "$@"
