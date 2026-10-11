#!/bin/bash
# Double-click this file to build (if needed) and run Eden.
# Safe to run repeatedly - only rebuilds what actually changed.

set -e

# Move to this script's own folder, no matter where it's been moved to,
# and no matter if the path has spaces in it.
cd "$(dirname "$0")"

echo "Eden - build & run"
echo "=================="
echo ""

# Self-heal a stale build folder. CMake refuses to reuse a build/ whose
# CMakeCache.txt was created in a different location ("current CMakeCache.txt
# directory is different than ..."): that happens when build/ is copied from
# another folder, the project folder is moved/renamed, or build output was
# ever committed to git and came back with a checkout. build/ is pure output,
# so just start it over.
if [ -f build/CMakeCache.txt ]; then
    CACHED_DIR=$(grep -m1 '^CMAKE_CACHEFILE_DIR:INTERNAL=' build/CMakeCache.txt | cut -d= -f2-)
    if [ "$CACHED_DIR" != "$(pwd -P)/build" ]; then
        echo "The build folder was created for a different location:"
        echo "  $CACHED_DIR"
        echo "Rebuilding it from scratch (one-time, a few minutes)..."
        rm -rf build
    fi
fi

mkdir -p build
cd build

echo "Configuring..."
# Release by default: Debug builds are several times slower on the CPU AND
# switch on Vulkan validation layers (see Renderer.cpp's NDEBUG check), which
# costs a lot of frame time. For debugging with validation, run:
#   EDEN_BUILD_TYPE=Debug ./"Run Eden (Mac).command"
BUILD_TYPE="${EDEN_BUILD_TYPE:-Release}"
echo "Build type: $BUILD_TYPE"
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" > /tmp/eden_cmake.log 2>&1 || {
    echo ""
    echo "CMake configuration failed. Full log:"
    cat /tmp/eden_cmake.log
    echo ""
    read -p "Press Enter to close..."
    exit 1
}

echo "Building..."
make > /tmp/eden_make.log 2>&1 || {
    echo ""
    echo "Build failed. Full log:"
    cat /tmp/eden_make.log
    echo ""
    read -p "Press Enter to close..."
    exit 1
}

echo ""
echo "Build succeeded. Launching Eden..."
echo ""

./Eden

echo ""
echo "Eden closed."

# Back to the project root (we're inside build/) - .git lives there,
# not in build/.
cd ..

if [ ! -d ".git" ]; then
    echo ""
    echo "Not connected to GitHub yet - run \"Connect Eden to GitHub"
    echo "(Mac).command\" once to set that up, then this prompt will"
    echo "appear here every time Eden closes."
    read -p "Press Enter to close this window..."
    exit 0
fi

# Shared safety helpers (restore remote-only files instead of deleting them on
# GitHub, refuse >100 MB files, explain push errors).
if [ ! -f "Tools/eden_git_safety.sh" ]; then
    echo "Tools/eden_git_safety.sh is missing - re-extract the project zip."
    read -p "Press Enter to close this window..."
    exit 1
fi
source "Tools/eden_git_safety.sh"

# Always asks, even with nothing new to push (matches how this was
# asked for) - a plain "git push" with nothing pending just reports
# "Everything up-to-date" and exits 0, so that case is cheap to allow
# through rather than trying to pre-detect and skip the dialog.
osascript -e 'display dialog "Push local changes to GitHub (main)?" buttons {"Not now", "Push"} default button "Push"' > /tmp/eden_pushdialog_$$.log 2>&1
if ! grep -q "Push" /tmp/eden_pushdialog_$$.log; then
    rm -f /tmp/eden_pushdialog_$$.log
    exit 0
fi
rm -f /tmp/eden_pushdialog_$$.log

# See "Connect Eden to GitHub (Mac).command"'s comment on why this is
# `if ! osascript ...` rather than a standalone call followed by a `$?`
# check - a literal "Cancel" button makes osascript exit nonzero, which
# would otherwise trip this script's `set -e` and abort silently.
if ! osascript -e 'text returned of (display dialog "GitHub Personal Access Token:" default answer "" with hidden answer buttons {"Cancel", "Push"} default button "Push")' > /tmp/eden_pat_$$.txt 2>/tmp/eden_pat_cancelled_$$.txt; then
    rm -f /tmp/eden_pat_$$.txt /tmp/eden_pat_cancelled_$$.txt
    exit 0
fi
GH_TOKEN=$(cat /tmp/eden_pat_$$.txt)
rm -f /tmp/eden_pat_$$.txt /tmp/eden_pat_cancelled_$$.txt

eden_untrack_build_output   # first: build/ etc. must leave the repo, not be 'kept'
eden_protect_remote_only_files

git add -A
git reset -q -- ':(glob).eden_askpass*' 2>/dev/null || true
if ! eden_check_large_staged_files; then
    read -p "Press Enter to close this window..."
    exit 1
fi
if ! git diff --cached --quiet; then
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M")
    git commit -m "Eden session $TIMESTAMP" > /dev/null
fi

# Ephemeral GIT_ASKPASS relay, same shape as "Connect Eden to GitHub
# (Mac).command" - deleted immediately after use, credential.helper
# explicitly disabled for this one command so nothing gets cached to
# Keychain, token never written anywhere durable. See that script's
# comment for the full reasoning.
# Temp file OUTSIDE the project folder: it holds the token in plaintext while
# this runs and must never be reachable by `git add -A`.
ASKPASS_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/eden_askpass.XXXXXX")"
cat > "$ASKPASS_SCRIPT" << EOF
#!/bin/bash
echo "$GH_TOKEN"
EOF
chmod +x "$ASKPASS_SCRIPT"

if GIT_ASKPASS="$ASKPASS_SCRIPT" git -c credential.helper= push origin main 2>/tmp/eden_push_err_$$.log; then
    rm -f "$ASKPASS_SCRIPT" /tmp/eden_push_err_$$.log
    unset GH_TOKEN
    osascript -e 'display dialog "Pushed to GitHub." buttons {"OK"} default button "OK"' > /dev/null 2>&1
else
    ERR=$(cat /tmp/eden_push_err_$$.log)
    rm -f "$ASKPASS_SCRIPT" /tmp/eden_push_err_$$.log
    unset GH_TOKEN
    echo "Push failed:"
    echo "$ERR"
    HINT=$(eden_explain_push_error "$ERR")
    if [ -n "$HINT" ]; then
        echo ""
        echo "What this probably means: $HINT"
    fi
    osascript -e "display dialog \"Push failed:\n\n$(echo "$ERR" | sed 's/"/\\"/g' | head -c 500)\" buttons {\"OK\"} default button \"OK\" with icon stop" > /dev/null 2>&1
fi

read -p "Press Enter to close this window..."
