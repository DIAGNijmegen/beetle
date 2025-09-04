#!/bin/bash

# Determine the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Navigate to the project root directory
cd "$SCRIPT_DIR/../../.."

docker run --rm \
    -v "$(pwd)":/beetle \
    -w /beetle/code/ \
    --network host \
    --entrypoint bash \
    --gpus all \
    beetle \
    -c "python3 inference.py"