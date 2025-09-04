#!/bin/bash
set -e

# This script downloads and extracts the BEETLE dataset.
# Ensure you have curl and unzip installed on your system.
# In total, the dataset is 151 GB in size, so downloading with a 100 Mbps connection should take about 3.5 hours.

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

DATA_DIR="./data"
DOWNLOAD_URL="https://zenodo.org/api/records/16812932/files-archive"
ARCHIVE_NAME="beetle.zip"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "Downloading data from $DOWNLOAD_URL..."
curl -L -o "$ARCHIVE_NAME" "$DOWNLOAD_URL"

# Extract the main archive
echo "Extracting $ARCHIVE_NAME..."
unzip -o "$ARCHIVE_NAME"
rm "$ARCHIVE_NAME"

# Extract sub-archives
for archive in images.zip annotations.zip model.zip; do
    if [ -f "$archive" ]; then
        echo "Extracting $archive..."
        unzip -o "$archive"
        rm "$archive"
    else
        echo "Warning: $archive not found, skipping..."
    fi
done

echo "Download and extraction complete."