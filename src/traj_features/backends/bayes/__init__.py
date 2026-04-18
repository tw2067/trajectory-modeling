"""Bayesian backend package.

Keep imports lazy so loading sibling modules (e.g. ``bayes.classify``)
does not eagerly import PyMC/PyTensor-heavy code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .model import BayesianTrajPS, BayesConfig

__all__ = ["BayesianTrajPS", "BayesConfig"]


def __getattr__(name: str):
	if name in {"BayesianTrajPS", "BayesConfig"}:
		from .model import BayesianTrajPS, BayesConfig
		return {"BayesianTrajPS": BayesianTrajPS, "BayesConfig": BayesConfig}[name]
	raise AttributeError(name)