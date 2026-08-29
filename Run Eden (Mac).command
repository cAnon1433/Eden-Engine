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

mkdir -p build
cd build

echo "Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Debug > /tmp/eden_cmake.log 2>&1 || {
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

git add -A
if ! git diff --cached --quiet; then
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M")
    git commit -m "Eden session $TIMESTAMP" > /dev/null
fi

# Ephemeral GIT_ASKPASS relay, same shape as "Connect Eden to GitHub
# (Mac).command" - deleted immediately after use, credential.helper
# explicitly disabled for this one command so nothing gets cached to
# Keychain, token never written anywhere durable. See that script's
# comment for the full reasoning.
ASKPASS_SCRIPT="$(pwd)/.eden_askpass_$$.sh"
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
    osascript -e "display dialog \"Push failed:\n\n$(echo "$ERR" | sed 's/"/\\"/g' | head -c 500)\" buttons {\"OK\"} default button \"OK\" with icon stop" > /dev/null 2>&1
fi

read -p "Press Enter to close this window..."
