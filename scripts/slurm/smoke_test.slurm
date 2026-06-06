#!/usr/bin/env bash
#SBATCH --job-name=smoke-test
#SBATCH --output=logs/outs/smoke_%j.out
#SBATCH --error=logs/errs/smoke_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-shamir

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
while [[ ! -f "$REPO_ROOT/pyproject.toml" && "$REPO_ROOT" != "/" ]]; do
	REPO_ROOT="$(dirname "$REPO_ROOT")"
done
[[ -f "$REPO_ROOT/pyproject.toml" ]] || { echo "Could not locate repo root"; exit 1; }

cd "$REPO_ROOT" || exit 1
pytest -q "$REPO_ROOT/tests/test_smoke_backends.py" -s

EXIT_CODE=$?

# Attempt to clean PyTensor cache (best-effort)
if [ -f "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" ]; then
	bash "$REPO_ROOT/scripts/slurm/cleanup_pytensor_cache.sh" || true
fi

exit $EXIT_CODE