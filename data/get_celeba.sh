#!/usr/bin/env bash
# Shell script to download the CelebA dataset (aligned & cropped images).
# Maintains a similar structure to the CIFAR-10 download script.
# It checks for an existing dataset and only downloads if not present, then extracts the zip.

# Exit immediately if a command exits with a non-zero status (optional safety)
set -e  # Exit on any error

# Ensure the script is running from its own directory (for relative paths to work)
DIR="$( cd "$( dirname "$0" )" && pwd -P )"
cd "$DIR"

# Target directory for the CelebA dataset
DATASET_DIR="data/celeba"

# Check if the dataset is already downloaded and extracted
if [ -d "$DATASET_DIR" ] && [ -n "$(ls -A "$DATASET_DIR")" ]; then
    echo "CelebA dataset already exists in $DATASET_DIR. Skipping download."
    exit 0
fi

# Create the dataset directory if it does not exist
mkdir -p "$DATASET_DIR"

# URL of the CelebA dataset (aligned & cropped images)
URL="https://archive.org/download/celeba/Img/img_align_celeba.zip"

echo "Downloading CelebA dataset..."
# Download the dataset zip file
if ! wget --no-check-certificate -O "$DATASET_DIR/img_align_celeba.zip" "$URL"; then
    echo "Error: Failed to download CelebA dataset."
    exit 1
fi

echo "Extracting the dataset..."
# Extract the zip file into the dataset directory
if ! unzip -q "$DATASET_DIR/img_align_celeba.zip" -d "$DATASET_DIR"; then
    echo "Error: Failed to extract CelebA dataset."
    exit 1
fi

echo "Done. CelebA dataset is available in $DATASET_DIR."
