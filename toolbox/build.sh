#!/bin/bash
# Build the halo-forge toolbox image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="halo-forge"
IMAGE_TAG="latest"

echo "Building halo-forge toolbox..."
echo "Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo

podman build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile .

echo
echo "Build complete!"
echo
echo "To create the toolbox:"
echo "  toolbox create halo-forge --image localhost/${IMAGE_NAME}:${IMAGE_TAG}"
echo
echo "To enter the toolbox:"
echo "  toolbox enter halo-forge"

