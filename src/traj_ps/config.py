# src/traj_ps/config.py
from __future__ import annotations
import os, yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List


def load_yaml(path: str | None, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Load YAML if it exists; otherwise return fallback dict copy."""
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return dict(fallback)

@dataclass
class DataDefaults:
    seed: int = 920
    n_patients: int = 120
    bin_width: float = 1/12  # monthly
    embed_features: tuple[str,...] = ("eGFR","HbA1c")
    agg_features:   tuple[str,...] = ("SBP","MedA")

def load_configs(data_cfg: str | None, train_cfg: str | None,
                 data_fallback: Dict[str, Any] | None = None,
                 train_fallback: Dict[str, Any] | None = None):
    data = load_yaml(data_cfg, fallback=(data_fallback or asdict(DataDefaults())))
    train = load_yaml(train_cfg, fallback=(train_fallback or {}))
    return data, train

@dataclass
class FeatureConfig:
    """Configuration for a single biomarker/feature."""
    name: str  # 'eGFR', 'CD4_count', 'MMSE'
    unit: str = ""

    # Feature direction and interpretation
    direction: str = "decreasing"
    higher_better: bool = True  # False for features like viral_load

    # Feature type
    feature_type: str = "continuous"  # "continuous", "binary", "count"
    has_periodicity: bool = False  # True for features with periodic variation (e.g., blood pressure)
    
    # Trajectory classification thresholds
    flat_threshold: float = -1.0
    decline_threshold: float = -2.0
    nonlinear_gap: float = 3.0
    
    # Feature direction
    higher_better: bool = True  # False for features like viral_load
    
    # Normal range
    normal_min: float = 60.0
    normal_max: float = 120.0
    critical_threshold: float = 15.0


@dataclass
class DiseaseConfig:
    """Disease-specific configuration."""
    name: str  # 'ckd', 'hiv', 'alzheimers'
    
    # Primary trajectory feature
    primary_feature: FeatureConfig
    
    # Additional covariates
    covariate_names: List[str] = field(default_factory=list)
    secondary_features: List[FeatureConfig] = field(default_factory=list)
    
    # Trajectory type labels
    trajectory_types: List[str] = field(default_factory=lambda: [
        'prolonged_nonprogression',
        'linear_decline',
        'nonlinear'
    ])

    trajectory_type_map: Dict[str, str] = field(default_factory=lambda: {
        'nonprogression': 'prolonged_nonprogression',
        'linear': 'linear_decline',
        'nonlinear': 'nonlinear'
    })
    
    @classmethod
    def for_ckd(cls):
        return cls(
            name='ckd',
            primary_feature=FeatureConfig(
                name='eGFR',
                unit='mL/min/1.73m²',
                direction='decreasing',
                higher_better=True,
                feature_type='continuous',
                has_periodicity=False,
                flat_threshold=-1.0,
                decline_threshold=-2.0,
                nonlinear_gap=3.0,
                normal_min=60.0,
                normal_max=120.0,
                critical_threshold=15.0,
            ),
            covariate_names=['Creatinine', 'SBP', 'DBP', 'HbA1c', 'MedA'],
            secondary_features=[
                FeatureConfig(
                    name='Creatinine',
                    unit='mg/dL',
                    direction='increasing',
                    higher_better=False,  # Higher creatinine = worse kidney function
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=0.7,
                    normal_max=1.3,
                    critical_threshold=3.0,
                ),
                FeatureConfig(
                    name='SBP',
                    unit='mmHg',
                    direction='stable',
                    higher_better=False,  # High BP is bad
                    feature_type='continuous',
                    has_periodicity=True,  # Blood pressure varies
                    normal_min=90,
                    normal_max=140,
                    critical_threshold=180,
                ),
                FeatureConfig(
                    name='DBP',
                    unit='mmHg',
                    direction='stable',
                    higher_better=False,
                    feature_type='continuous',
                    has_periodicity=True,
                    normal_min=60,
                    normal_max=90,
                    critical_threshold=110,
                ),
                FeatureConfig(
                    name='HbA1c',
                    unit='%',
                    direction='stable',
                    higher_better=False,  # High HbA1c = poor glucose control
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=4.0,
                    normal_max=5.7,
                    critical_threshold=9.0,
                ),
                FeatureConfig(
                    name='MedA',
                    unit='',
                    direction='stable',
                    higher_better=True,  # Medication adherence
                    feature_type='binary',
                    has_periodicity=False,
                ),
            ],
            trajectory_types=['prolonged_nonprogression', 'linear_decline', 'nonlinear'],
            trajectory_type_map={
                'nonprogression': 'prolonged_nonprogression',
                'linear': 'linear_decline',
                'nonlinear': 'nonlinear'
            }
        )
    
    @classmethod
    def for_hiv(cls):
        return cls(
            name='hiv',
            primary_feature=FeatureConfig(
                name='CD4_count',
                unit='cells/μL',
                direction='increasing',
                higher_better=True,  # Higher CD4 = better immune function
                feature_type='continuous',
                has_periodicity=False,
                flat_threshold=5.0,
                decline_threshold=20.0,
                nonlinear_gap=50.0,
                normal_min=500.0,
                normal_max=1500.0,
                critical_threshold=200.0,
            ),
            covariate_names=['viral_load', 'weight', 'hemoglobin', 'ART'],
            secondary_features=[
                FeatureConfig(
                    name='viral_load',
                    unit='copies/mL',
                    direction='decreasing',
                    higher_better=False,  # Lower viral load = better
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=0.0,
                    normal_max=50.0,  # Undetectable
                    critical_threshold=100000.0,
                ),
                FeatureConfig(
                    name='weight',
                    unit='kg',
                    direction='stable',
                    higher_better=True,  # Weight gain often good in HIV
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=50.0,
                    normal_max=90.0,
                    critical_threshold=40.0,
                ),
                FeatureConfig(
                    name='hemoglobin',
                    unit='g/dL',
                    direction='stable',
                    higher_better=True,  # Higher Hb = better
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=12.0,
                    normal_max=16.0,
                    critical_threshold=8.0,
                ),
                FeatureConfig(
                    name='ART',
                    unit='',
                    direction='stable',
                    higher_better=True,  # Being on ART is good
                    feature_type='binary',
                    has_periodicity=False,
                ),
            ],
            trajectory_types=['stable', 'gradual_decline', 'rapid_decline'],
            trajectory_type_map={
                'nonprogression': 'stable',
                'linear': 'gradual_decline',
                'nonlinear': 'rapid_decline'
            }
        )
    
    @classmethod
    def for_alzheimers(cls):
        return cls(
            name='alzheimers',
            primary_feature=FeatureConfig(
                name='MMSE',
                unit='score',
                direction='decreasing',
                higher_better=True,  # Higher MMSE = better cognition
                feature_type='continuous',
                has_periodicity=False,
                flat_threshold=-0.5,
                decline_threshold=-2.0,
                nonlinear_gap=3.0,
                normal_min=24.0,
                normal_max=30.0,
                critical_threshold=10.0,
            ),
            covariate_names=['ADAS_cog', 'hippocampal_volume', 'CSF_tau', 'medications'],
            secondary_features=[
                FeatureConfig(
                    name='ADAS_cog',
                    unit='score',
                    direction='increasing',
                    higher_better=False,  # Higher ADAS-cog = worse cognition
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=0.0,
                    normal_max=10.0,
                    critical_threshold=30.0,
                ),
                FeatureConfig(
                    name='hippocampal_volume',
                    unit='cm³',
                    direction='decreasing',
                    higher_better=True,  # Larger hippocampus = better
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=3.0,
                    normal_max=4.5,
                    critical_threshold=2.0,
                ),
                FeatureConfig(
                    name='CSF_tau',
                    unit='pg/mL',
                    direction='increasing',
                    higher_better=False,  # Higher tau = worse (pathology marker)
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=0.0,
                    normal_max=300.0,
                    critical_threshold=600.0,
                ),
                FeatureConfig(
                    name='medications',
                    unit='',
                    direction='stable',
                    higher_better=True,  # Being on cholinesterase inhibitors
                    feature_type='binary',
                    has_periodicity=False,
                ),
            ],
            trajectory_types=['stable', 'slow_decline', 'rapid_decline'],
            trajectory_type_map={
                'nonprogression': 'stable',
                'linear': 'slow_decline',
                'nonlinear': 'rapid_decline'
            }
        )

    @classmethod
    def for_parkinsons(cls):
        """Parkinson's disease with MDS-UPDRS-III (increasing = worsening)."""
        return cls(
            name='parkinsons',
            primary_feature=FeatureConfig(
                name='MDS_UPDRS_III',
                unit='score',
                direction='increasing',  # Scores increase as disease worsens
                higher_better=False,  # Higher scores = worse motor symptoms
                feature_type='continuous',
                has_periodicity=False,
                flat_threshold=0.5,  # Minimal increase
                decline_threshold=2.0,  # Rapid worsening (increase > 2 points/year)
                nonlinear_gap=3.0,
                normal_min=0.0,
                normal_max=10.0,  # Normal elderly baseline
                critical_threshold=50.0,  # Severe impairment
            ),
            covariate_names=['LEDD', 'tremor_score', 'rigidity_score', 'medications'],
            secondary_features=[
                FeatureConfig(
                    name='LEDD',
                    unit='mg/day',
                    direction='increasing',
                    higher_better=False,  # Higher LEDD = more medication needed (worse disease)
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=0.0,
                    normal_max=600.0,
                    critical_threshold=1500.0,
                ),
                FeatureConfig(
                    name='tremor_score',
                    unit='score',
                    direction='increasing',
                    higher_better=False,  # Higher tremor = worse
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=0.0,
                    normal_max=1.0,
                    critical_threshold=3.0,
                ),
                FeatureConfig(
                    name='rigidity_score',
                    unit='score',
                    direction='increasing',
                    higher_better=False,  # Higher rigidity = worse
                    feature_type='continuous',
                    has_periodicity=False,
                    normal_min=0.0,
                    normal_max=1.0,
                    critical_threshold=3.0,
                ),
                FeatureConfig(
                    name='medications',
                    unit='',
                    direction='stable',
                    higher_better=True,  # Being on PD medications (levodopa, etc.)
                    feature_type='binary',
                    has_periodicity=False,
                ),
            ],
            trajectory_types=['stable', 'slow_progression', 'rapid_progression'],
            trajectory_type_map={
                'nonprogression': 'stable',
                'linear': 'slow_progression',
                'nonlinear': 'rapid_progression'
            }
        )