#!/bin/bash
#
# Double-clickable macOS launcher for GestureBloom.
#
# Finder runs .command files from the user's home directory, not from where the
# file lives, so the project path is derived from this script's own location
# rather than hardcoded. That keeps the launcher working for anyone who clones
# the repo to a different path -- a hardcoded /Users/someone/... would make it
# useless to every other person.
#
# Setup (once):
#     chmod +x scripts/GestureBloom.command
#
# Then double-click it in Finder, or make a Desktop alias:
#     right-click -> Make Alias -> drag the alias to the Desktop
#
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR" || {
    echo "Could not enter project directory: $PROJECT_DIR"
    read -r -n 1 -p "Press any key to close..."
    exit 1
}

printf '\033[1mGestureBloom\033[0m\n'
echo "Project: $PROJECT_DIR"
echo

# ---- virtual environment ------------------------------------------------- #
# Prefer .venv, fall back to .venv312 -- some setups need an older Python for
# mediapipe or torch wheels, and this launcher should find either.
VENV=""
for candidate in .venv .venv312 .venv311; do
    if [ -f "$candidate/bin/activate" ]; then
        VENV="$candidate"
        break
    fi
done

if [ -z "$VENV" ]; then
    cat <<'MSG'
No virtual environment found.

Set one up first:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e '.[dev,render,live]'
MSG
    read -r -n 1 -p "Press any key to close..."
    exit 1
fi

# shellcheck disable=SC1090,SC1091
source "$VENV/bin/activate"
echo "Environment: $VENV  ($(python --version 2>&1))"

# ---- is the package installed? ------------------------------------------- #
if ! python -c "import gesturebloom" 2>/dev/null; then
    cat <<'MSG'

The gesturebloom package is not installed in this environment.

    pip install -e '.[dev,render,live]'
MSG
    read -r -n 1 -p "Press any key to close..."
    exit 1
fi

# ---- calibration is optional but strongly preferred ---------------------- #
ARGS=()
if [ -f "calibration.json" ]; then
    ARGS+=(--calibration calibration.json)
    echo "Calibration: calibration.json"
else
    echo "Calibration: none (run 'gesturebloom calibrate --out calibration.json' for a better feel)"
fi

cat <<'KEYS'

Keys:  ESC/Q quit  |  SPACE pause  |  R reset  |  S screenshot
       C camera feed  |  K skeleton

Left hand controls grow. Right hand controls bloom. Show both hands.

KEYS

python -m gesturebloom.cli run "${ARGS[@]}"
STATUS=$?

# Finder closes the Terminal window the instant the script ends, taking any
# error message with it. Pause on failure so the message is actually readable --
# this is the difference between "it didn't work" and a diagnosable problem.
if [ $STATUS -ne 0 ]; then
    echo
    echo "Exited with status $STATUS."
    echo "If this is a camera issue: System Settings > Privacy & Security > Camera"
    echo "and enable Terminal, then fully quit and reopen Terminal."
    echo
    read -r -n 1 -p "Press any key to close..."
fi

exit $STATUS
