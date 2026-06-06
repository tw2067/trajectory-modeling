#!/usr/bin/env bash
# Remove PyTensor cache for this SLURM job and prune old cache dirs.
# Safe: only deletes per-job dirs and cache dirs older than 7 days.

set -euo pipefail

HOME_DIR=${HOME:-$(eval echo ~${USER:-$(whoami)})}
PYTENSOR_CACHE_DIR="$HOME_DIR/.pytensor_cache"

echo "[cleanup_pytensor_cache] running cleanup for job ${SLURM_JOB_ID:-unknown}"

if [ -d "$PYTENSOR_CACHE_DIR" ]; then
    # remove cache directory that matches this job id
    if [ -n "${SLURM_JOB_ID:-}" ]; then
        TARGET="$PYTENSOR_CACHE_DIR/${SLURM_JOB_ID}"
        if [ -d "$TARGET" ]; then
            echo "[cleanup_pytensor_cache] removing $TARGET"
            rm -rf "$TARGET" || true
        fi
    fi

    # prune cache dirs older than 7 days to avoid accumulating lots of artifacts
    echo "[cleanup_pytensor_cache] pruning cache dirs older than 7 days in $PYTENSOR_CACHE_DIR"
    find "$PYTENSOR_CACHE_DIR" -maxdepth 1 -mindepth 1 -type d -mtime +7 -print -exec rm -rf {} + || true
fi

echo "[cleanup_pytensor_cache] done"
