#!/usr/bin/env python3
"""
Qualitative Validation Figure — Trajectory Modeling Framework

Produces two outputs per run:
  1. Per-biomarker figures (3 columns × 2 overlaid patients) showing raw EHR
     measurements for high-confidence trajectory archetype windows.
  2. A single overview figure (6 rows × 3 columns) with 2 patients overlaid
     per panel, clear column headers, and biomarker row labels.

Outputs: results/figures/qualitative_validation/

Usage:
    python scripts/figures/qualitative_validation.py
    python scripts/figures/qualitative_validation.py --threshold 0.85 --seed 7
    python scripts/figures/qualitative_validation.py --biomarkers lactate creatinine
"""

from __future__ import annotations

import argparse
import os
import random
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_ROOT = os.environ.get("PHYSIONET_ROOT", "/home/gaga/data/physionet")
OUTPUT_DIR = _REPO_ROOT / "results" / "figures" / "qualitative_validation"

# ─── Quality filters ──────────────────────────────────────────────────────────
#
# For the Stable class, require that the relative range of raw values in the
# window is small (values don't swing more than MAX_STABLE_REL_RANGE × mean).
# For directional classes (Increase/Decline), require a minimum normalised
# trend: the linear slope × window_length must exceed MIN_TREND_STRENGTH × mean.
# Both filters have a relaxed fallback so we always find examples.

MIN_OBS_IN_WINDOW    = 3      # minimum raw measurements for a usable panel
HIGH_CONF_THRESHOLD  = 0.75
N_EXAMPLES_PER_CLASS = 2
MAX_STABLE_REL_RANGE = 0.30   # (max−min)/mean < this for a "flat" trajectory
MIN_TREND_STRENGTH   = 0.15   # |slope×window|/mean > this for a clear trend

# ─── Biomarker Configuration ──────────────────────────────────────────────────

BIOMARKER_CONFIGS: dict[str, dict] = {
    "lactate": {
        "display_name": "Lactate",
        "unit": "mmol/L",
        "prob_file": f"{_DATA_ROOT}/eicu/sepsis/lactate_trajectory_probs_bayes.csv",
        "ts_file":   f"{_DATA_ROOT}/eicu/sepsis/lactate_timeseries.csv",
        "patient_id_col": "stay_id",
        "value_col":   "lactate",
        "time_col":    "time_days",
        "window_col":  "time_day",
        "window_days": 3,
        "prob_cols": {
            "Stable":           "lactate_stable",
            "Gradual Increase": "lactate_gradual",
            "Non-linear Increase":   "lactate_rapid",
        },
        "direction": {"Stable": 0, "Gradual Increase": +1, "Non-linear Increase": +1},
        "clinical_threshold": 2.0,
        "threshold_label":    "Normal limit (2 mmol/L)",
        "threshold_style":    "above",
    },
    "wbc": {
        "display_name": "WBC",
        "unit": "× 10³/μL",
        "prob_file": f"{_DATA_ROOT}/eicu/sepsis/wbc_trajectory_probs_bayes.csv",
        "ts_file":   f"{_DATA_ROOT}/eicu/sepsis/wbc_timeseries.csv",
        "patient_id_col": "stay_id",
        "value_col":   "wbc",
        "time_col":    "time_days",
        "window_col":  "time_day",
        "window_days": 3,
        "prob_cols": {
            "Stable":           "wbc_stable",
            "Gradual Increase": "wbc_gradual",
            "Non-linear Increase":   "wbc_rapid",
        },
        "direction": {"Stable": 0, "Gradual Increase": +1, "Non-linear Increase": +1},
        "clinical_threshold": 11.0,
        "threshold_label":    "Upper normal (11 × 10³/μL)",
        "threshold_style":    "above",
    },
    "platelets": {
        "display_name": "Platelets",
        "unit": "× 10³/μL",
        "prob_file": f"{_DATA_ROOT}/eicu/sepsis/platelet_trajectory_probs_bayes.csv",
        "ts_file":   f"{_DATA_ROOT}/eicu/sepsis/platelets_timeseries.csv",
        "patient_id_col": "stay_id",
        "value_col":   "platelet",
        "time_col":    "time_days",
        "window_col":  "time_day",
        "window_days": 3,
        "prob_cols": {
            "Stable":          "platelet_stable",
            "Gradual Decline": "platelet_gradual",
            "Non-linear Decline":   "platelet_rapid",
        },
        "direction": {"Stable": 0, "Gradual Decline": -1, "Non-linear Decline": -1},
        "clinical_threshold": 150.0,
        "threshold_label":    "Thrombocytopenia (<150 × 10³/μL)",
        "threshold_style":    "below",
    },
    "creatinine": {
        "display_name": "Creatinine",
        "unit": "mg/dL",
        "prob_file": f"{_DATA_ROOT}/eicu/aki/aki_trajectory_probs.csv",
        "ts_file":   f"{_DATA_ROOT}/eicu/aki/creatinine_timeseries.csv",
        "patient_id_col": "stay_id",
        "value_col":   "creatinine",
        "time_col":    "time_days",
        "window_col":  "time_day",
        "window_days": 7,
        "prob_cols": {
            "Stable":           "prob_stable",
            "Gradual Increase": "prob_gradual_increase",
            "Non-linear Increase":   "prob_rapid_increase",
        },
        "direction": {"Stable": 0, "Gradual Increase": +1, "Non-linear Increase": +1},
        "clinical_threshold": 1.2,
        "threshold_label":    "Upper normal (1.2 mg/dL)",
        "threshold_style":    "above",
    },
    "bilirubin": {
        "display_name": "Bilirubin",
        "unit": "mg/dL",
        # Bayesian merged file — prob values repeat across same-day measurements;
        # load_probs deduplicates by (stay_id, time_day).
        "prob_file": f"{_DATA_ROOT}/eicu/liver/liver_trajectory_probs_bayes.csv",
        "ts_file":   f"{_DATA_ROOT}/eicu/liver/bilirubin_timeseries.csv",
        "patient_id_col": "stay_id",
        "value_col":   "bilirubin",
        "time_col":    "time_days",
        "window_col":  "time_day",
        "window_days": 5,
        "prob_cols": {
            "Stable":              "prob_stable",
            "Gradual Increase":    "prob_gradual_increase",
            "Non-linear Increase": "prob_rapid_increase",
        },
        "direction": {"Stable": 0, "Gradual Increase": +1, "Non-linear Increase": +1},
        "clinical_threshold": 1.2,
        "threshold_label":    "Upper normal (1.2 mg/dL)",
        "threshold_style":    "above",
    },
    "pf_ratio": {
        "display_name": "P/F Ratio",
        "unit": "mmHg",
        # Bayesian merged file — "improvement" = increasing P/F ratio (better oxygenation).
        "prob_file": f"{_DATA_ROOT}/eicu/ventilator/ventilator_trajectory_probs_bayes.csv",
        "ts_file":   f"{_DATA_ROOT}/eicu/ventilator/pf_ratio_timeseries.csv",
        "patient_id_col": "stay_id",
        "value_col":   "pf_ratio",
        "time_col":    "time_days",
        "window_col":  "time_day",
        "window_days": 3,
        "prob_cols": {
            "Stable":                 "prob_stable",
            "Gradual Improvement":    "prob_gradual_improvement",
            "Non-linear Improvement": "prob_rapid_improvement",
        },
        "direction": {"Stable": 0, "Gradual Improvement": +1, "Non-linear Improvement": +1},
        "clinical_threshold": 300.0,
        "threshold_label":    "ARDS threshold (<300 mmHg)",
        "threshold_style":    "below",
    },
}

# Archetype colour scheme — consistent across all biomarkers
CLASS_PALETTE: dict[str, str] = {
    "Stable":           "#2196F3",   # blue
    "Gradual Increase": "#FF9800",   # orange
    "Non-linear Increase":     "#F44336",   # red
    "Gradual Decline":         "#FF9800",
    "Non-linear Decline":      "#F44336",
    "Gradual Change":          "#FF9800",
    "Non-linear Change":       "#F44336",
    "Gradual Improvement":     "#FF9800",
    "Non-linear Improvement":  "#F44336",
}

# Markers/styles for the two overlaid patients
_PATIENT_STYLES = [
    {"marker": "o",  "linestyle": "--",  "alpha_scatter": 0.95, "alpha_line": 0.50, "label": "Patient 1", "filled": True},
    {"marker": "^",  "linestyle": ":",   "alpha_scatter": 0.80, "alpha_line": 0.35, "label": "Patient 2", "filled": False},
]

# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_probs(cfg: dict) -> pd.DataFrame:
    prob_cols = list(cfg["prob_cols"].values())
    pid_col   = cfg["patient_id_col"]
    usecols   = [pid_col, "time_day"] + prob_cols

    df = pd.read_csv(cfg["prob_file"])
    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{cfg['prob_file']} missing columns: {missing}\nAvailable: {list(df.columns)}"
        )
    df = df[usecols].copy()
    df = df.dropna(subset=prob_cols, how="all")
    # Some Bayesian merged files repeat prob values across same-day measurements
    df = df.drop_duplicates(subset=[pid_col, "time_day"])
    return df


def load_timeseries(cfg: dict) -> pd.DataFrame:
    pid_col    = cfg["patient_id_col"]
    val_col    = cfg["value_col"]
    time_col   = cfg["time_col"]
    window_col = cfg["window_col"]

    df   = pd.read_csv(cfg["ts_file"])
    keep = list(dict.fromkeys(c for c in [pid_col, val_col, time_col, window_col] if c in df.columns))
    df   = df[keep].copy()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df   = df.dropna(subset=[val_col])
    return df


# ─── Window Extraction ────────────────────────────────────────────────────────

def get_window_measurements(
    ts_df: pd.DataFrame,
    patient_id: int | float,
    time_day:   int | float,
    cfg: dict,
) -> pd.DataFrame:
    """
    Reproduce the pipeline's windowing_col='time_day' logic:
        window = [time_day − window_days, time_day]  (integer day comparison)
    Returns measurements with an added rel_time column (0 = window end).
    """
    pid_col    = cfg["patient_id_col"]
    time_col   = cfg["time_col"]
    window_col = cfg["window_col"]
    W          = cfg["window_days"]

    pat  = ts_df[ts_df[pid_col] == patient_id].copy()
    mask = (pat[window_col] >= time_day - W) & (pat[window_col] <= time_day)
    win  = pat[mask].copy()
    win["rel_time"] = win[time_col] - time_day
    return win.sort_values("rel_time")


# ─── Example Selection ────────────────────────────────────────────────────────

def _quality_checks(
    x: np.ndarray,
    y: np.ndarray,
    window_days: float,
    expected_dir: int,
    check_direction: bool,
    check_trend:     bool,
    check_stable:    bool,
) -> bool:
    """Return True if this window passes all requested quality filters."""
    if len(x) < 2:
        return not (check_direction or check_stable or check_trend)

    slope, *_ = stats.linregress(x, y)
    mean_abs   = float(np.mean(np.abs(y))) + 1e-9
    norm_trend = slope * window_days / mean_abs
    rel_range  = (float(y.max()) - float(y.min())) / mean_abs

    if check_direction and expected_dir != 0:
        if expected_dir > 0 and slope <= 0:
            return False
        if expected_dir < 0 and slope >= 0:
            return False

    if check_trend and expected_dir != 0:
        if abs(norm_trend) < MIN_TREND_STRENGTH:
            return False

    if check_stable and expected_dir == 0:
        if rel_range > MAX_STABLE_REL_RANGE:
            return False

    return True


def find_representative_examples(
    probs_df: pd.DataFrame,
    ts_df:    pd.DataFrame,
    cfg:      dict,
    threshold:    float = HIGH_CONF_THRESHOLD,
    n_per_class:  int   = N_EXAMPLES_PER_CLASS,
    rng: random.Random | None = None,
) -> dict[str, list[tuple]]:
    """
    For each trajectory class, find patient windows where:
      • That class has the highest probability AND
      • Its probability exceeds `threshold` AND
      • At least MIN_OBS_IN_WINDOW raw measurements exist AND
      • Raw measurements visually match the class direction/flatness.

    Falls back through progressively relaxed filters (strict → direction-only →
    no quality filter), lower probability thresholds (0.70 → 0.65 → 0.60 → 0.30),
    and finally removes the dominance requirement for classes that are rarely
    dominant (e.g. P/F "Gradual Change").
    """
    pid_col       = cfg["patient_id_col"]
    val_col       = cfg["value_col"]
    window_days   = cfg["window_days"]
    prob_col_map  = cfg["prob_cols"]
    direction_map = cfg["direction"]
    all_prob_cols = list(prob_col_map.values())

    probs = probs_df.copy()
    probs["_dominant"] = probs[all_prob_cols].idxmax(axis=1)

    results: dict[str, list[tuple]] = {}

    for class_label, prob_col in prob_col_map.items():
        expected_dir = direction_map.get(class_label, 0)

        # Stage A: require dominance, try decreasing thresholds
        candidates = pd.DataFrame()
        for thresh in [threshold, 0.70, 0.65, 0.60]:
            mask = (probs["_dominant"] == prob_col) & (probs[prob_col] > thresh)
            cand = probs[mask].sort_values(prob_col, ascending=False)
            cand = cand.drop_duplicates(subset=[pid_col], keep="first")
            if not cand.empty:
                if thresh < threshold:
                    print(f"    [{cfg['display_name']}] '{class_label}': "
                          f"lowered threshold to {thresh:.2f}")
                candidates = cand
                break

        # Stage B: if still empty, drop dominance requirement — use any window
        # where this class has the highest raw probability available (last resort).
        if candidates.empty:
            best = (
                probs[probs[prob_col] > 0.10]
                .sort_values(prob_col, ascending=False)
                .drop_duplicates(subset=[pid_col], keep="first")
            )
            if not best.empty:
                print(f"    [{cfg['display_name']}] '{class_label}': "
                      f"dominance relaxed — best available p={best[prob_col].max():.2f}")
                candidates = best
            else:
                print(f"    [{cfg['display_name']}] '{class_label}': no candidates found")
                results[class_label] = []
                continue

        # Progressively relaxed quality filter stages
        filter_stages = [
            dict(min_obs=MIN_OBS_IN_WINDOW, check_direction=True,  check_trend=True,  check_stable=True),
            dict(min_obs=MIN_OBS_IN_WINDOW, check_direction=True,  check_trend=False, check_stable=True),
            dict(min_obs=MIN_OBS_IN_WINDOW, check_direction=True,  check_trend=False, check_stable=False),
            dict(min_obs=2,                 check_direction=False, check_trend=False, check_stable=False),
        ]

        selected: list[tuple] = []
        seed_val = rng.randint(0, 99999) if rng else 42

        for stage_idx, filt in enumerate(filter_stages):
            shuffled = candidates.sample(frac=1, random_state=seed_val + stage_idx)
            for _, row in shuffled.iterrows():
                pid = row[pid_col]
                td  = row["time_day"]
                p   = row[prob_col]

                win = get_window_measurements(ts_df, pid, td, cfg)
                if len(win) < filt["min_obs"]:
                    continue

                passes = _quality_checks(
                    win["rel_time"].values,
                    win[val_col].values,
                    window_days,
                    expected_dir,
                    filt["check_direction"],
                    filt["check_trend"],
                    filt["check_stable"],
                )
                if not passes:
                    continue

                if any(s[0] == pid for s in selected):
                    continue

                selected.append((pid, td, p))
                if len(selected) >= n_per_class:
                    break

            if selected:
                break

        results[class_label] = selected
        n = len(selected)
        print(f"    [{cfg['display_name']}] '{class_label}': {n} example(s) selected")

    return results


# ─── Single-panel drawing helper ──────────────────────────────────────────────

def _draw_panel(
    ax: plt.Axes,
    windows: list[tuple],
    ts_df:   pd.DataFrame,
    cfg:     dict,
    class_label: str,
    show_xlabel: bool = False,
    show_ylabel: bool = True,
    show_legend: bool = True,
    subtitle:    str  = "",
) -> None:
    """
    Draw up to 2 patients overlaid on `ax` with distinct markers/linestyles.
    Clinical threshold is drawn as a dashed horizontal line.
    """
    val_col     = cfg["value_col"]
    window_days = cfg["window_days"]
    threshold   = cfg.get("clinical_threshold")
    thr_label   = cfg.get("threshold_label", "")
    thr_style   = cfg.get("threshold_style", "above")
    color       = CLASS_PALETTE.get(class_label, "#888888")

    if not windows:
        ax.set_facecolor("#F5F5F5")
        ax.text(0.5, 0.52, "Not observed as\ndominant archetype\nin this cohort",
                ha="center", va="center", transform=ax.transAxes,
                color="#9E9E9E", fontsize=8, style="italic", linespacing=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#BDBDBD")
        if subtitle:
            ax.set_title(subtitle, fontsize=8.5, pad=4, color="#757575")
        return

    probs_shown: list[float] = []

    for win_idx, (patient_id, time_day, prob) in enumerate(windows[:2]):
        style = _PATIENT_STYLES[win_idx]
        win   = get_window_measurements(ts_df, patient_id, time_day, cfg)
        if win.empty:
            continue

        x = win["rel_time"].values
        y = win[val_col].values
        probs_shown.append(prob)

        fc = color if style["filled"] else "white"
        ec = "white" if style["filled"] else color

        ax.plot(x, y, color=color, alpha=style["alpha_line"],
                linestyle=style["linestyle"], linewidth=1.3, zorder=2)
        ax.scatter(x, y, s=48, zorder=4,
                   color=color if style["filled"] else "none",
                   facecolors=fc,
                   edgecolors=color,
                   linewidths=1.4 if not style["filled"] else 0.5,
                   marker=style["marker"],
                   alpha=style["alpha_scatter"])

    # Clinical threshold
    if threshold is not None:
        ax.axhline(threshold, color="#c62828", linestyle="--",
                   linewidth=0.9, alpha=0.65, zorder=1, label=thr_label)

    ax.axvline(0, color="black", linewidth=0.6, alpha=0.18, zorder=1)
    ax.set_xlim(-window_days - 0.3, 1.2)
    ax.tick_params(labelsize=7.5)
    sns.despine(ax=ax)

    if show_xlabel:
        ax.set_xlabel("Days to window end", fontsize=8)
    else:
        ax.set_xlabel("")

    if show_ylabel:
        ax.set_ylabel(f"{cfg['display_name']}\n({cfg['unit']})", fontsize=8)
    else:
        ax.set_ylabel("")

    # Legend for markers
    if show_legend and len(probs_shown) > 0:
        handles = []
        for i, p in enumerate(probs_shown):
            s = _PATIENT_STYLES[i]
            h = plt.Line2D(
                [], [],
                marker=s["marker"],
                color=color,
                linestyle=s["linestyle"],
                markerfacecolor=color if s["filled"] else "white",
                markeredgecolor=color,
                markersize=5,
                alpha=s["alpha_scatter"],
                label=f"Patient {i+1}  p={p:.2f}",
            )
            handles.append(h)
        if threshold is not None:
            handles.append(mpatches.Patch(
                color="#c62828", alpha=0.6, label=thr_label
            ))
        ax.legend(handles=handles, fontsize=6.0, loc="best",
                  framealpha=0.55, handlelength=1.5,
                  borderpad=0.4, labelspacing=0.3)

    if subtitle:
        ax.set_title(subtitle, fontsize=8.5, pad=4)


# ─── Per-biomarker Figure ────────────────────────────────────────────────────

def plot_biomarker_figure(
    cfg:      dict,
    examples: dict[str, list[tuple]],
    ts_df:    pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
) -> Path:
    """
    One figure per biomarker: 1 row × 3 columns, 2 patients overlaid per panel.
    """
    class_labels = list(cfg["prob_cols"].keys())
    n_cols = len(class_labels)

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 4.4), squeeze=False)

    fig.suptitle(
        f"Qualitative Validation — {cfg['display_name']}",
        fontsize=13, fontweight="bold", y=1.03,
    )

    for col_idx, class_label in enumerate(class_labels):
        ax = axes[0, col_idx]
        _draw_panel(
            ax,
            windows=examples.get(class_label, []),
            ts_df=ts_df,
            cfg=cfg,
            class_label=class_label,
            show_xlabel=True,
            show_ylabel=(col_idx == 0),
            show_legend=True,
            subtitle=class_label,
        )

    plt.tight_layout()
    out_path = output_dir / f"{cfg['value_col']}_validation.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─── Overview Figure ─────────────────────────────────────────────────────────

def plot_overview_figure(
    all_examples: dict[str, dict],
    all_ts:       dict[str, pd.DataFrame],
    all_cfgs:     dict[str, dict],
    output_dir:   Path,
    dpi: int = 300,
) -> Path:
    """
    Single overview: 6 rows (biomarkers) × 3 columns (archetypes).
    Each panel overlays 2 patients, showing 2 archetype examples side-by-side.
    Column headers and row labels make the figure self-explanatory.
    """
    biomarkers = list(all_cfgs.keys())
    n_rows = len(biomarkers)
    n_cols = 3   # always Stable | Gradual | Rapid

    col_titles = [
        "Stable / Non-Progression",
        "Gradual Change",
        "Non-linear Change",
    ]

    sns.set_theme(style="whitegrid", font_scale=0.9)

    # Use GridSpec: first row is a header strip, remaining rows are data panels
    fig = plt.figure(figsize=(15, 4.0 * n_rows + 1.2))
    gs = fig.add_gridspec(
        n_rows + 1, n_cols,
        height_ratios=[0.12] + [1.0] * n_rows,
        hspace=0.48,
        wspace=0.30,
        top=0.97, bottom=0.04, left=0.07, right=0.98,
    )

    # ── Column header row ────────────────────────────────────────────────────
    for col_idx, title in enumerate(col_titles):
        ax_h = fig.add_subplot(gs[0, col_idx])
        ax_h.set_facecolor("#ECEFF1")
        ax_h.text(
            0.5, 0.5, title,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            transform=ax_h.transAxes,
        )
        ax_h.set_xticks([])
        ax_h.set_yticks([])
        for spine in ax_h.spines.values():
            spine.set_visible(False)

    # ── Data panels ──────────────────────────────────────────────────────────
    axes: list[list[plt.Axes]] = []
    for row_idx in range(n_rows):
        row_axes = []
        for col_idx in range(n_cols):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])
            row_axes.append(ax)
        axes.append(row_axes)

    for row_idx, bm_key in enumerate(biomarkers):
        cfg      = all_cfgs[bm_key]
        ts_df    = all_ts[bm_key]
        examples = all_examples[bm_key]
        class_labels = list(cfg["prob_cols"].keys())

        # Left spine gets the biomarker label
        is_bottom = (row_idx == n_rows - 1)

        for col_idx in range(n_cols):
            ax          = axes[row_idx][col_idx]
            class_label = class_labels[col_idx] if col_idx < len(class_labels) else None

            if class_label is None:
                ax.set_visible(False)
                continue

            _draw_panel(
                ax,
                windows=examples.get(class_label, []),
                ts_df=ts_df,
                cfg=cfg,
                class_label=class_label,
                show_xlabel=is_bottom,
                show_ylabel=(col_idx == 0),
                show_legend=True,
                subtitle=class_label,   # biomarker-specific label (e.g. "Gradual Decline")
            )

            # Biomarker row label on the leftmost column
            if col_idx == 0:
                ax.set_ylabel(f"{cfg['display_name']}\n({cfg['unit']})",
                              fontsize=8.5, labelpad=6)

    out_path = output_dir / "overview_all_biomarkers.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--threshold", type=float, default=HIGH_CONF_THRESHOLD,
                   help=f"Min probability for high-confidence windows (default: {HIGH_CONF_THRESHOLD})")
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES_PER_CLASS,
                   help=f"Patients per class per panel (default: {N_EXAMPLES_PER_CLASS})")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--dpi",  type=int, default=300, help="Output DPI (default: 300)")
    p.add_argument(
        "--biomarkers", nargs="+",
        default=list(BIOMARKER_CONFIGS.keys()),
        choices=list(BIOMARKER_CONFIGS.keys()),
        help="Subset of biomarkers to process (default: all)",
    )
    p.add_argument("--no-overview", action="store_true",
                   help="Skip the combined overview figure")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng  = random.Random(args.seed)
    np.random.seed(args.seed)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output → {OUTPUT_DIR}\n")

    all_examples: dict[str, dict]          = {}
    all_ts:       dict[str, pd.DataFrame]  = {}
    saved:        list[Path]               = []

    for bm_key in args.biomarkers:
        cfg = BIOMARKER_CONFIGS[bm_key]
        print(f"{'='*58}")
        print(f"  {cfg['display_name']}")
        print(f"{'='*58}")

        prob_path = Path(cfg["prob_file"])
        ts_path   = Path(cfg["ts_file"])

        if not prob_path.exists():
            print(f"  [SKIP] Not found: {prob_path}")
            continue
        if not ts_path.exists():
            print(f"  [SKIP] Not found: {ts_path}")
            continue

        probs_df = load_probs(cfg)
        ts_df    = load_timeseries(cfg)
        print(f"  Probs: {len(probs_df):,} rows | "
              f"TS: {len(ts_df):,} rows ({ts_df[cfg['patient_id_col']].nunique():,} patients)")

        print("  Selecting examples …")
        examples = find_representative_examples(
            probs_df, ts_df, cfg,
            threshold=args.threshold,
            n_per_class=args.n_examples,
            rng=rng,
        )

        print("  Plotting …")
        out = plot_biomarker_figure(cfg, examples, ts_df, OUTPUT_DIR, dpi=args.dpi)
        print(f"  → {out.relative_to(_REPO_ROOT)}\n")
        saved.append(out)

        all_examples[bm_key] = examples
        all_ts[bm_key]       = ts_df

    if not args.no_overview and all_examples:
        print(f"{'='*58}")
        print("  Overview (all biomarkers)")
        print(f"{'='*58}")
        ov_cfgs = {k: BIOMARKER_CONFIGS[k] for k in all_examples}
        ov_path = plot_overview_figure(all_examples, all_ts, ov_cfgs, OUTPUT_DIR, dpi=args.dpi)
        print(f"  → {ov_path.relative_to(_REPO_ROOT)}\n")
        saved.append(ov_path)

    print(f"Done. {len(saved)} figure(s) saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
