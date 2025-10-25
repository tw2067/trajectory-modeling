import os
import pytest
import importlib.util

# Keep CI stable/fast
os.environ["TORCH_NO_JIT_PROFILE"] = "1"
os.environ.setdefault("PYTHONHASHSEED", "920")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None

# Use pytest.importorskip (safer than homegrown flags)
def require_or_skip(pkg, reason):
    try:
        return __import__(pkg)
    except Exception:
        pytest.skip(reason)

has_torch = _has("torch")
has_pymc  = _has("pymc")
has_pygam = _has("pygam")

skip_if_no_torch = pytest.mark.skipif(not has_torch, reason="torch not installed (deep extra)")
skip_if_no_pymc  = pytest.mark.skipif(not has_pymc,  reason="pymc not installed (bayes extra)")
skip_if_no_pygam = pytest.mark.skipif(not has_pygam, reason="pygam not installed (gam extra)")