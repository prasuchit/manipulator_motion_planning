#!/usr/bin/env bash

set -e

TARGET_DIR=${1:-.}

echo "🧹 Removing Python cache directories..."
find "$TARGET_DIR" -type d -name "__pycache__" -exec rm -rf {} +
find "$TARGET_DIR" -type f -name "*.pyc" -delete
