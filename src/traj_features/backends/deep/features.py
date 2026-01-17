import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.linear_model import LinearRegression


def extract_deep_trajectory_features(
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    window_size: float = 1.0,
    min_obs: int = 3,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract per-window linear trajectory features.
    
    This is a PROXY for the deep learning model during validation.
    For actual deployment, this should use the trained LSTM/Transformer model.
    
    For each patient:
    1. Divide time series into windows
    2. Fit linear regression within each window
    3. Extract slope and intercept per window
    4. Aggregate to patient-level features
    
    Parameters
    ----------
    dynamic_df : pd.DataFrame
        Columns: pid, time, feature_name, value
    feature : str
        Primary feature to model
    window_size : float
        Window size in years (default: 1.0)
    min_obs : int
        Minimum observations per window (default: 3)
    
    Returns
    -------
    features_df : pd.DataFrame
        Columns: pid, {feature}_slope, {feature}_intercept
    trajectories_df : pd.DataFrame
        Columns: pid, time, predicted_value (reconstructed trajectories)
    """
    print(f"[Deep] Extracting windowed trajectory features for {feature}")
    print(f"[Deep] Window size: {window_size} years, min_obs: {min_obs}")
    
    # Filter to primary feature
    feat_df = dynamic_df[dynamic_df['feature_name'] == feature].copy()
    
    if feat_df.empty:
        print(f"[WARNING] No data for feature {feature}")
        return pd.DataFrame(columns=['pid', f'{feature}_slope', f'{feature}_intercept']), \
               pd.DataFrame(columns=['pid', 'time', 'predicted_value'])
    
    # Determine overall time range
    t_min = feat_df['time'].min()
    t_max = feat_df['time'].max()
    
    # Create windows
    n_windows = int(np.ceil((t_max - t_min) / window_size))
    windows = [(t_min + i * window_size, t_min + (i + 1) * window_size) 
               for i in range(n_windows)]
    
    print(f"[Deep] Time range: [{t_min:.2f}, {t_max:.2f}], n_windows: {n_windows}")
    
    features_list = []
    trajectories_list = []
    
    for pid in feat_df['pid'].unique():
        pid_data = feat_df[feat_df['pid'] == pid].sort_values('time')
        
        window_slopes = []
        
        for win_start, win_end in windows:
            # Get observations in this window
            win_mask = (pid_data['time'] >= win_start) & (pid_data['time'] < win_end)
            win_data = pid_data[win_mask]
            
            if len(win_data) < min_obs:
                continue
            
            times = win_data['time'].values
            values = win_data['value'].values
            
            # Fit linear regression
            try:
                lr = LinearRegression()
                lr.fit(times.reshape(-1, 1), values)
                
                slope = lr.coef_[0]
                window_slopes.append(slope)
                
                # Generate predictions for reconstruction
                pred_times = np.linspace(win_start, win_end, 20)
                pred_values = lr.predict(pred_times.reshape(-1, 1))
                
                for t, v in zip(pred_times, pred_values):
                    trajectories_list.append({
                        'pid': pid,
                        'time': t,
                        'predicted_value': v
                    })
                    
            except Exception as e:
                print(f"[ERROR] Linear fit failed for pid={pid}, window=[{win_start:.2f}, {win_end:.2f}]: {e}")
                continue
        
        # Aggregate: mean slope across windows, first value as baseline
        if window_slopes:
            mean_slope = np.mean(window_slopes)
            baseline = pid_data['value'].iloc[0]
            
            features_list.append({
                'pid': pid,
                f'{feature}_slope': mean_slope,
                f'{feature}_intercept': baseline,
                'n_windows': len(window_slopes)
            })
    
    features_df = pd.DataFrame(features_list)
    trajectories_df = pd.DataFrame(trajectories_list)
    
    print(f"[Deep] Extracted {len(features_df)} patient features from {n_windows} windows")
    if not features_df.empty:
        print(f"[Deep] Mean slope: {features_df[f'{feature}_slope'].mean():.4f} ± {features_df[f'{feature}_slope'].std():.4f}")
    
    return features_df, trajectories_df