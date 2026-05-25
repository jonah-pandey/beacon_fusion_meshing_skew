#!/bin/bash
set -e

# Define descriptive, explicit directory targets
KLIPPER_DIRECTORY="${HOME}/klipper"
KLIPPER_ENVIRONMENT="${HOME}/klippy-env"
IS_CREALITY_OS=0

# Detect Creality OS / Buildroot embedded environments
if grep -Fqs "ID=buildroot" /etc/os-release; then
    KLIPPER_DIRECTORY="/usr/data/klipper"
    KLIPPER_ENVIRONMENT="/usr/share/klippy-env"
    IS_CREALITY_OS=1
fi

# Verify physical directory existence paths before executing destructive operations
if [ ! -d "$KLIPPER_DIRECTORY" ] || [ ! -d "$KLIPPER_ENVIRONMENT" ]; then
    echo "Installation Aborted: Klipper repository or python virtual environment directory could not be located."
    echo "Paths Evaluated -> Repository: $KLIPPER_DIRECTORY | Environment: $KLIPPER_ENVIRONMENT"
    exit 1
fi

# Resolve the absolute path of the script directory regardless of where it was called from
SCRIPT_SOURCE_DIRECTORY="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
TARGET_MODULE_FILE="spatial_geometry_engine.py"

echo "System Detection -> Creality OS Embedded: $IS_CREALITY_OS"
echo "Source Path: ${SCRIPT_SOURCE_DIRECTORY}/${TARGET_MODULE_FILE}"

# Enforce mathematical module dependencies within the isolated Klippy python environment
echo "Verifying scientific and linear algebra runtime dependencies within virtual environment..."
if ! "$KLIPPER_ENVIRONMENT/bin/python" -c "import numpy; import scipy" 2>/dev/null; then
    echo "Injecting missing numpy and scipy dependencies into Klippy virtual environment..."
    "$KLIPPER_ENVIRONMENT/bin/pip" install --upgrade pip
    "$KLIPPER_ENVIRONMENT/bin/pip" install numpy scipy
fi

# Execute module symlinking loop
if [ -e "${SCRIPT_SOURCE_DIRECTORY}/${TARGET_MODULE_FILE}" ]; then
    TARGET_LINK_PATH="${KLIPPER_DIRECTORY}/klippy/extras/${TARGET_MODULE_FILE}"
    
    # Safely clear stale references or old installation files if they exist
    if [ -L "$TARGET_LINK_PATH" ] || [ -e "$TARGET_LINK_PATH" ]; then
        echo "Clearing legacy module reference path..."
        rm -f "$TARGET_LINK_PATH"
    fi
    
    echo "Establishing symbolic link inside Klipper extras directory..."
    ln -s "${SCRIPT_SOURCE_DIRECTORY}/${TARGET_MODULE_FILE}" "$TARGET_LINK_PATH"
    
    # Protect the upstream Git tree tracking array from local modifications
    GIT_EXCLUDE_FILE="${KLIPPER_DIRECTORY}/.git/info/exclude"
    if [ -d "${KLIPPER_DIRECTORY}/.git" ]; then
        if ! grep -q "klippy/extras/${TARGET_MODULE_FILE}" "$GIT_EXCLUDE_FILE"; then
            echo "Registering custom extension inside .git/info/exclude tracking filter..."
            echo "klippy/extras/${TARGET_MODULE_FILE}" >> "$GIT_EXCLUDE_FILE"
        fi
    fi
else
    echo "Installation Failure: Target source module '${TARGET_MODULE_FILE}' missing from script folder path."
    exit 1
fi

# Automatically recycle the host service layers to bind the new code configurations
echo "Spatially Varying Kinematics Engine deployed successfully. Restarting Klipper service daemon..."
if [ $IS_CREALITY_OS -eq 1 ]; then
    killall python 2>/dev/null || true
    echo "Klipper process recycled inside Creality OS environment."
else
    sudo systemctl restart klipper
    echo "Systemd Klipper service recycled cleanly."
fi