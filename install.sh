#!/bin/bash
set -e

KLIPPER_DIRECTORY="${HOME}/klipper"
KLIPPER_ENVIRONMENT="${HOME}/klippy-env"
IS_CREALITY_OS=0

if grep -Fqs "ID=buildroot" /etc/os-release; then
    KLIPPER_DIRECTORY="/usr/data/klipper"
    KLIPPER_ENVIRONMENT="/usr/share/klippy-env"
    IS_CREALITY_OS=1
fi

if [ ! -d "$KLIPPER_DIRECTORY" ] || [ ! -d "$KLIPPER_ENVIRONMENT" ]; then
    echo "Installation Aborted: Klipper path definitions could not be validated."
    exit 1
fi

SCRIPT_SOURCE_DIRECTORY="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
TARGET_MODULE_FILE="beacon_meshing_skew.py"

echo "Syncing repository tracking branch state..."
cd "$SCRIPT_SOURCE_DIRECTORY"
if [ -d ".git" ]; then
    git fetch origin main
    git stash -q || true
    git merge origin/main --ff-only
    git stash pop -q || true
fi

echo "Verifying runtime dependencies inside virtual environment..."
if ! "$KLIPPER_ENVIRONMENT/bin/python" -c "import numpy; import scipy" 2>/dev/null; then
    echo "Installing missing scientific packages into Klippy virtual environment..."
    "$KLIPPER_ENVIRONMENT/bin/pip" install --upgrade pip
    "$KLIPPER_ENVIRONMENT/bin/pip" install numpy scipy
fi

if [ -e "${SCRIPT_SOURCE_DIRECTORY}/${TARGET_MODULE_FILE}" ]; then
    TARGET_LINK_PATH="${KLIPPER_DIRECTORY}/klippy/extras/${TARGET_MODULE_FILE}"
    
    if [ -L "$TARGET_LINK_PATH" ] || [ -e "$TARGET_LINK_PATH" ]; then
        rm -f "$TARGET_LINK_PATH"
    fi
    
    echo "Creating symbolic link inside Klipper extras directory..."
    ln -s "${SCRIPT_SOURCE_DIRECTORY}/${TARGET_MODULE_FILE}" "$TARGET_LINK_PATH"
    
    GIT_EXCLUDE_FILE="${KLIPPER_DIRECTORY}/.git/info/exclude"
    if [ -d "$KLIPPER_DIRECTORY/.git" ]; then
        if ! grep -q "klippy/extras/${TARGET_MODULE_FILE}" "$GIT_EXCLUDE_FILE"; then
            echo "klippy/extras/${TARGET_MODULE_FILE}" >> "$GIT_EXCLUDE_FILE"
        fi
    fi
else
    echo "Installation Failure: Target extension file missing from execution folder path."
    exit 1
fi

echo "Extension deployed. Cycling Klipper host services..."
if [ $IS_CREALITY_OS -eq 1 ]; then
    killall python 2>/dev/null || true
else
    sudo systemctl restart klipper
fi
echo "Update loop completed successfully."