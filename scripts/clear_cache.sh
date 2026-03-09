#!/usr/bin/env bash

set -e

TARGET_DIR=${1:-.}

echo "🧹 Removing Python cache directories..."
# Remove __pycache__ dirs
find "$TARGET_DIR" -type d -name "__pycache__" -print0 | xargs -0 rm -rf
# Remove .pyc files
find "$TARGET_DIR" -type f -name "*.pyc" -print0 | xargs -0 rm -f