#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Clearing all Python caches..."

# Remove all __pycache__ directories
find "$ROOT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove all .pyc files
find "$ROOT_DIR" -type f -name "*.pyc" -delete 2>/dev/null

# Remove all .pyo files
find "$ROOT_DIR" -type f -name "*.pyo" -delete 2>/dev/null

echo "Cache cleared!"