#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper.
# Preferred location: scripts/launchers/generate_launchers.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/launchers/generate_launchers.sh" "$@"
