#!/bin/bash

# Determine the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Navigate to the project root directory
cd "$SCRIPT_DIR/../../.."

docker run --rm -it \
    -v "$(pwd)":/beetle \
    -w /beetle/code/ \
    --gpus all \
    beetle
