#!/usr/bin/env bash

set -e

TARGET_DIR=${1:-.}

echo "🔍 Running Ruff linting + autofix..."
ruff check "$TARGET_DIR" --fix

echo "📦 Sorting imports with isort..."
isort "$TARGET_DIR"

echo "🖤 Formatting with Black..."
black "$TARGET_DIR"

echo "✅ Python formatting complete and caches cleared."