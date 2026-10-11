#!/bin/bash
# Shared safety helpers for "Connect Eden to GitHub (Mac).command" and
# "Run Eden (Mac).command" - source this, don't run it.
#
# Why this exists: Eden's project folder is re-created from a zip, so it
# starts WITHOUT .git/, .gitignore and .github/ . A bare `git add -A` on such
# a folder (a) stages files that exist on GitHub but not here as DELETIONS -
# including .github/workflows/* , whose deletion GitHub refuses unless the
# token has the "workflow" scope, and which would also silently destroy the
# publish action - and (b) has no .gitignore to keep build output and
# generated files out of the commit.

# Remove BUILD OUTPUT that is already tracked by the repo. .gitignore only
# stops NEW files being added; a file committed once (e.g. by an older version
# of these scripts, before a .gitignore existed) stays tracked and keeps being
# re-committed on every push. Tracked build folders are actively harmful: a
# CMakeCache.txt from one machine breaks every other checkout ("current
# CMakeCache.txt directory is different than ..."), including GitHub Actions.
# Only well-known build-output folders are touched, and only in the repo
# (git rm --cached) - nothing on disk is deleted.
eden_untrack_build_output() {
    local tracked count
    tracked=$(git ls-files -ci --exclude-standard | grep -E '^(build|build-[^/]*|dist|out|cmake-build-[^/]*)/' || true)
    [ -z "$tracked" ] && return 0
    count=$(printf '%s\n' "$tracked" | wc -l | tr -d ' ')
    echo ""
    echo "Removing $count build-output file(s) from the repository (they should never have been committed; .gitignore now covers them)."
    printf '%s\n' "$tracked" | while IFS= read -r f; do
        git rm --cached -q --ignore-unmatch -- "$f" 2>/dev/null || true
    done
}

# Restore (instead of deleting on GitHub) files GitHub has but this folder
# doesn't. Dot-files/dot-folders (.github, .gitignore, ...) are ALWAYS kept -
# a fresh unzip never contains them, so "missing" never means "deleted on
# purpose". For anything else, ask once.
eden_protect_remote_only_files() {
    local missing count answer f
    missing=$(git ls-files -d)
    [ -z "$missing" ] && return 0

    # Always keep dot-paths.
    while IFS= read -r f; do
        case "$f" in
            .*) git checkout -- "$f" 2>/dev/null || true ;;
        esac
    done <<< "$missing"

    missing=$(git ls-files -d)
    [ -z "$missing" ] && return 0
    count=$(printf '%s\n' "$missing" | wc -l | tr -d ' ')

    echo ""
    echo "$count file(s) exist on GitHub but not in this folder, e.g.:"
    printf '%s\n' "$missing" | head -5
    answer=$(osascript -e "display dialog \"$count file(s) are on GitHub but missing from this folder (for example because the folder was re-created from a zip).\n\nKeep them on GitHub (recommended), or delete them from GitHub?\" buttons {\"Delete from GitHub\", \"Keep on GitHub\"} default button \"Keep on GitHub\"" 2>/dev/null || true)
    if printf '%s' "$answer" | grep -q "Delete from GitHub"; then
        echo "OK - those files will be deleted from GitHub."
    else
        echo "Keeping them on GitHub."
        while IFS= read -r f; do
            git checkout -- "$f" 2>/dev/null || true
        done <<< "$missing"
    fi
}

# GitHub rejects any single file over 100 MB - but only after uploading
# everything else. Catch it before the commit instead. Returns 1 if found.
eden_check_large_staged_files() {
    local f size big=""
    while IFS= read -r f; do
        [ -f "$f" ] || continue
        size=$(wc -c < "$f" | tr -d ' ')
        if [ "$size" -gt 95000000 ]; then
            big="$big\n  $f ($((size / 1048576)) MB)"
        fi
    done < <(git diff --cached --name-only)
    if [ -n "$big" ]; then
        echo ""
        echo "Not pushing: these staged files are over GitHub's 100 MB limit:"
        printf "$big\n"
        osascript -e "display dialog \"Not pushed - files over GitHub's 100 MB limit are staged:\n$(printf "$big" | head -c 300)\n\nAdd them to .gitignore (or delete them) and run again.\" buttons {\"OK\"} default button \"OK\" with icon stop" > /dev/null 2>&1
        return 1
    fi
    return 0
}

# Turn git's raw push errors into something actionable. Prints a hint (or
# nothing if the error isn't recognized).
eden_explain_push_error() {
    local err="$1"
    if printf '%s' "$err" | grep -qi "without .workflow. scope\|workflow.*scope"; then
        echo "Your token can't change files under .github/workflows. Create the token with the 'workflow' scope (classic token) or 'Workflows: Read and write' permission (fine-grained)."
    elif printf '%s' "$err" | grep -qi "secret\|push protection\|GH013"; then
        echo "GitHub blocked the push because it found a secret (like a token) in the commit. Look at the file names in the message above, remove them from the commit, and revoke any token that was exposed."
    elif printf '%s' "$err" | grep -qi "authentication failed\|invalid username or password\|bad credentials\|could not read Password"; then
        echo "GitHub rejected the token: it may have expired, been revoked, or be mistyped. Create a new Personal Access Token and try again."
    elif printf '%s' "$err" | grep -qi "403\|permission to .* denied\|write access"; then
        echo "The token is valid but isn't allowed to write to this repo. For a fine-grained token, grant it 'Contents: Read and write' on Eden-Engine."
    elif printf '%s' "$err" | grep -qi "fetch first\|non-fast-forward\|rejected.*main"; then
        echo "GitHub has newer commits than this folder (for example made on github.com or by an Action). Run \"Connect Eden to GitHub (Mac).command\" again to re-align, then push."
    elif printf '%s' "$err" | grep -qi "100 MB\|large file\|pack exceeds"; then
        echo "A file is over GitHub's size limit. Make sure build output (build/, dist/) is in .gitignore."
    elif printf '%s' "$err" | grep -qi "could not resolve host\|unable to access\|timed out\|network"; then
        echo "Couldn't reach GitHub - check your internet connection and try again."
    fi
}
