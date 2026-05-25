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
    echo "Installation Aborted: Klipper repository or python environment directory not found."
    echo "Paths checked -> Repo: $KLIPPER_DIRECTORY | Env: $KLIPPER_ENVIRONMENT"
    exit 1
fi

SCRIPT_SOURCE_DIRECTORY="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
TARGET_MODULE_FILE="beacon_meshing_skew.py"

echo "System Status -> Creality OS Embedded: $IS_CREALITY_OS"
echo "Source File Path: ${SCRIPT_SOURCE_DIRECTORY}/${TARGET_MODULE_FILE}"

echo "Checking runtime dependencies inside virtual environment..."
if ! "$KLIPPER_ENVIRONMENT/bin/python" -c "import numpy; import scipy" 2>/dev/null; then
    echo "Installing missing dependencies (numpy, scipy) into Klippy virtual environment..."
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
    if [ -d "${KLIPPER_DIRECTORY}/.git" ]; then
        if ! grep -q "klippy/extras/${TARGET_MODULE_FILE}" "$GIT_EXCLUDE_FILE"; then
            echo "Registering module path inside .git/info/exclude filter..."
            echo "klippy/extras/${TARGET_MODULE_FILE}" >> "$GIT_EXCLUDE_FILE"
        fi
    fi
else
    echo "Installation Failure: Target file '${TARGET_MODULE_FILE}' was not found in the script's folder."
    exit 1
fi

echo "Module deployed successfully. Refreshing Klipper daemon..."
if [ $IS_CREALITY_OS -eq 1 ]; then
    killall python 2>/dev/null || true
else
    sudo systemctl restart klipper
fi
echo "Done."