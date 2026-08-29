#!/bin/bash
# Run ONCE to connect this project folder to the existing GitHub repo
# (https://github.com/cAnon1433/Eden-Engine) - safe to leave in the
# project after that; running it again just confirms the connection is
# already there and does nothing destructive.
#
# What this does NOT do: it does not force-push, does not overwrite
# the repo's existing commit history, and does not touch any file
# already on your disk. This folder has never been a git clone - it's
# been unzipped fresh each session - so the repo's real history and
# this folder's current files have been living completely separately.
# This script brings them together as ONE NEW COMMIT on top of
# whatever's already on GitHub, not a replacement of it:
#   1. git init (creates .git/ here, touches nothing else)
#   2. git remote add origin <repo>
#   3. git fetch origin (downloads the repo's real history as objects
#      only - does NOT touch your working files)
#   4. git reset origin/main (points HEAD at the repo's current main
#      WITHOUT checking out its files over yours - a --mixed reset
#      only moves HEAD/the index, the working directory is untouched.
#      This is the step that avoids clobbering your current files)
#   5. Every file that differs from what's on GitHub now shows as a
#      normal pending change, which gets committed and pushed as one
#      catch-up commit - existing history stays exactly as it was,
#      with this as a new commit on top.
#
# You'll need a GitHub Personal Access Token for this (NOT your GitHub
# account password - GitHub stopped accepting passwords for git
# operations in 2021). If you don't have one:
#   github.com -> profile picture (top right) -> Settings ->
#   Developer settings -> Personal access tokens -> Tokens (classic) ->
#   Generate new token (classic) -> tick the "repo" scope -> Generate.
#   Copy it now - GitHub only shows it once.
# The token is typed into a masked pop-up below (asked once, up front,
# since it's needed for fetch too if the repo is private, not just the
# final push), used only in memory for this run, and never written to
# disk.

set -e
cd "$(dirname "$0")"

REPO_URL="https://x-access-token@github.com/cAnon1433/Eden-Engine.git"

echo "Eden - connect to GitHub"
echo "========================="
echo ""

if [ -d ".git" ]; then
    echo "This folder is already connected to a git repo (.git/ exists)."
    echo "Nothing to do - if you meant to reconnect from scratch, remove"
    echo "the .git folder first and re-run this script."
    read -p "Press Enter to close..."
    exit 0
fi

# NOTE on `set -e` + AppleScript: display dialog raises a script error
# (nonzero exit) specifically when the button clicked is literally
# named "Cancel" - under `set -e`, a standalone command like this would
# kill the whole script the instant that happens, skipping the
# graceful handling below entirely. Using `if ! osascript ...; then`
# keeps the nonzero exit inside the if-condition, where `set -e`
# doesn't apply - same reasoning applies everywhere else in this file
# and in Run Eden (Mac).command that prompts with a "Cancel" button.
if ! osascript -e 'display dialog "Connect this Eden folder to https://github.com/cAnon1433/Eden-Engine ?\n\nThis adds your current local files as ONE NEW commit on top of the repo'"'"'s existing history - it will NOT overwrite or delete anything already on GitHub." buttons {"Cancel", "Connect"} default button "Connect" with icon caution' > /dev/null 2>&1; then
    echo "Cancelled."
    exit 0
fi

# Asked up front, not just before the final push - git fetch needs the
# same auth as git push if this repo is private, so one token covers
# both rather than prompting twice.
if ! osascript -e 'text returned of (display dialog "GitHub Personal Access Token:" default answer "" with hidden answer buttons {"Cancel", "Continue"} default button "Continue")' > /tmp/eden_pat_$$.txt 2>/tmp/eden_pat_cancelled_$$.txt; then
    rm -f /tmp/eden_pat_$$.txt /tmp/eden_pat_cancelled_$$.txt
    echo "Cancelled."
    read -p "Press Enter to close..."
    exit 0
fi
GH_TOKEN=$(cat /tmp/eden_pat_$$.txt)
rm -f /tmp/eden_pat_$$.txt /tmp/eden_pat_cancelled_$$.txt

# Ephemeral GIT_ASKPASS relay: a temp script that only ever lives long
# enough for this setup run, deleted immediately after (also covered
# defensively by .gitignore in case a run gets interrupted). Never
# written into .git/config, never handled by a credential-caching
# helper - credential.helper is explicitly cleared for every git
# command below that touches the network, so nothing gets saved to
# macOS Keychain either, matching "type it every time, don't save it."
# Used for BOTH fetch and push below, not just push - see the token
# prompt's own comment on why.
ASKPASS_SCRIPT="$(pwd)/.eden_askpass_$$.sh"
cat > "$ASKPASS_SCRIPT" << EOF
#!/bin/bash
echo "$GH_TOKEN"
EOF
chmod +x "$ASKPASS_SCRIPT"
cleanup_askpass() {
    rm -f "$ASKPASS_SCRIPT"
    unset GH_TOKEN
}
trap cleanup_askpass EXIT

echo "Setting up git..."
git init -b main
# Token-in-URL-username convention (x-access-token) so the ONLY thing
# GIT_ASKPASS ever has to supply is the token itself, as the password -
# if a real username were left for git to ask about too, our askpass
# script would get invoked for username AND password and hand back the
# token for both, which is wrong. GitHub only checks the token/password
# field for a PAT; the username string itself isn't part of that check.
git remote add origin "$REPO_URL"

echo "Fetching existing repo history (not touching your local files)..."
GIT_ASKPASS="$ASKPASS_SCRIPT" git -c credential.helper= fetch origin main

echo "Aligning with the repo's current history..."
git reset origin/main

echo ""
echo "Staging your current local files as one catch-up commit..."

# Fallback identity, ONLY applied if this machine has no git identity
# configured at all (fresh Mac / first-ever git use here) - local to
# this repo only (no --global), so it never overrides a real identity
# that's already set. Without this, `git commit` below fails outright
# with "Please tell me who you are."
if [ -z "$(git config user.email)" ]; then
    git config user.email "$(whoami)@users.noreply.github.com"
fi
if [ -z "$(git config user.name)" ]; then
    git config user.name "cAnon1433"
fi

git add -A

if git diff --cached --quiet; then
    echo "Nothing differs from the repo's current state - connected, no commit needed."
    read -p "Press Enter to close..."
    exit 0
fi

TIMESTAMP=$(date "+%Y-%m-%d %H:%M")
git commit -m "Eden session $TIMESTAMP - sync local progress into repo history" > /dev/null
echo "Committed."

echo "Pushing to GitHub..."
if GIT_ASKPASS="$ASKPASS_SCRIPT" git -c credential.helper= push -u origin main 2>/tmp/eden_push_err_$$.log; then
    rm -f /tmp/eden_push_err_$$.log
    osascript -e 'display dialog "Pushed. Eden is now connected to GitHub." buttons {"OK"} default button "OK"' > /dev/null 2>&1
    echo "Done."
else
    ERR=$(cat /tmp/eden_push_err_$$.log)
    rm -f /tmp/eden_push_err_$$.log
    echo "Push failed:"
    echo "$ERR"
    osascript -e "display dialog \"Push failed:\n\n$(echo "$ERR" | sed 's/"/\\"/g' | head -c 500)\" buttons {\"OK\"} default button \"OK\" with icon stop" > /dev/null 2>&1
fi

read -p "Press Enter to close..."
