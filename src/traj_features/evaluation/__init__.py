"""
Evaluation framework for comparing trajectory extraction methods.
"""

from .metrics import *
from .prediction import *

__all__ = [
    'compare_trajectory_assignments',
    'trajectory_agreement_score',
    'evaluate_downstream_prediction'
]
