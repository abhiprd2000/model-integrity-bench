#!/usr/bin/env bash
# Download the demo inputs (dog photo, ImageNet labels) into demo/assets/.
set -euo pipefail
d="$(cd "$(dirname "$0")" && pwd)/assets"; mkdir -p "$d"
base=https://raw.githubusercontent.com/pytorch/hub/master
curl -sfL -o "$d/dog.jpg" "$base/images/dog.jpg"
curl -sfL -o "$d/labels.txt" "$base/imagenet_classes.txt"
