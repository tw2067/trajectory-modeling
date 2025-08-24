import numpy as np


def flags_from_traj(traj, time_grid, flat_thr=-1, decline_thr=-2, nonlinear_gap=3):
    dt = np.diff(time_grid)
    dy = np.diff(traj)
    slopes = dy / dt

    # prolonged nonprogression: mostly flat and average slope of non-progression
    frac_flat = (slopes >= decline_thr).mean()
    total_flat = np.mean(slopes) >= flat_thr

    # linear decline: mostly steeply negative
    frac_decl = (slopes < decline_thr).mean()

    # nonlinear: mean slope of faster half vs slower half differs by > nonlinear_gap
    s = np.sort(slopes)                       # ascending (more negative first)
    half = len(s) // 2
    fast = s[:half].mean()                    # faster decline half (more negative)
    slow = s[half:].mean()                    # slower half
    nonlinear = abs(fast - slow) > nonlinear_gap

    return {
        'prolonged_nonprogression': (frac_flat >= 0.8) & total_flat,
        'linear_decline':            frac_decl >= 0.8,
        'nonlinear':                 nonlinear
    }


def pos_flags_from_traj(traj, time_grid, flat_thr=0.5, decline_thr=1, nonlinear_gap=2):
    dt = np.diff(time_grid)
    dy = np.diff(traj)
    slopes = dy / dt

    frac_flat = (slopes <= decline_thr).mean()
    total_flat = np.mean(slopes) <= flat_thr

    frac_decl = (slopes > decline_thr).mean()

    # nonlinear: mean slope of faster half vs slower half differs by > nonlinear_gap
    s = np.sort(slopes)
    half = len(s) // 2
    fast = s[half:].mean()                    # faster decline half (more positive)
    slow = s[:half].mean()                    # slower half
    nonlinear = abs(fast - slow) > nonlinear_gap

    return {
        'prolonged_nonprogression': (frac_flat >= 0.8) & total_flat,
        'linear_decline': frac_decl >= 0.8,
        'nonlinear': nonlinear
    }


def mmse_flag_from_traj(traj, time_grid, flat_thr=-0.3 ,decline_thr=-1.5, nonlinear_gap=1,
                        aging_thr = -0.5):
    dt = np.diff(time_grid)
    dy = np.diff(traj)
    slopes = dy / dt

    frac_flat = (slopes >= aging_thr).mean()
    total_flat = np.mean(slopes) >= flat_thr

    frac_slow = ((decline_thr <= slopes) & (slopes < aging_thr)).mean()
    frac_fast = (decline_thr > slopes).mean()

    s = np.sort(slopes)  # ascending (more negative first)
    half = len(s) // 2
    fast = s[:half].mean()  # faster decline half (more negative)
    slow = s[half:].mean()  # slower half
    nonlinear = abs(fast - slow) > nonlinear_gap

    return {
        'prolonged_nonprogression': (frac_flat >= 0.8) & total_flat,
        'slow_linear_decline': frac_slow >= 0.8,
        'fast_linear_decline': frac_fast >= 0.8,
        'nonlinear': nonlinear
    }
