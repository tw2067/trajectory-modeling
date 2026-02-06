# src/traj_ps/inference/export_embed.py
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, Optional
from pathlib import Path

def embedding_map_to_df(H_map: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows=[]
    for pid, H in H_map.items():
        T, D = H.shape
        for t in range(T):
            rows.append({"pid": pid, "t_idx": t, **{f"h{j+1}": float(H[t,j]) for j in range(D)}})
    return pd.DataFrame(rows)

def save_embeddings(H_map: Dict[str, np.ndarray], path: str) -> None:
    embedding_map_to_df(H_map).to_parquet(path, index=False)


def predict_trajectories(
    dynamic_df: pd.DataFrame,
    feature: str = "eGFR",
    model_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Use trained Deep model to predict trajectories.
    
    Parameters
    ----------
    dynamic_df : pd.DataFrame
        Longitudinal data with columns: pid, time, feature_name, value
    feature : str
        Feature to predict (e.g., 'eGFR', 'CD4_count', 'MMSE')
    model_path : str, optional
        Path to trained model checkpoint
    
    Returns
    -------
    pd.DataFrame
        Predicted trajectories with columns: pid, time, predicted_value
    """
    try:
        # Import Deep model components
        from traj_ps.backends.deep.model import GRUDModel
        import torch
        
        # Load model if path provided
        if model_path and Path(model_path).exists():
            model = torch.load(model_path)
            model.eval()
            print(f"  [Deep] Loaded model from {model_path}")
        else:
            raise ValueError("No trained model available")
        
        # Filter to feature
        feature_df = dynamic_df[dynamic_df['feature_name'] == feature].copy()
        
        if feature_df.empty:
            return pd.DataFrame(columns=['pid', 'time', 'predicted_value'])
        
        # Prepare data for model (patient-wise sequences)
        predictions = []
        
        for pid in feature_df['pid'].unique():
            patient_data = feature_df[feature_df['pid'] == pid].sort_values('time')
            
            times = patient_data['time'].values
            values = patient_data['value'].values
            
            # Create input tensor (shape: [seq_len, 1])
            X = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
            
            # Get predictions
            with torch.no_grad():
                y_pred = model(X.unsqueeze(0)).squeeze().cpu().numpy()
            
            # Store predictions
            for t, pred_val in zip(times, y_pred):
                predictions.append({
                    'pid': pid,
                    'time': t,
                    'predicted_value': float(pred_val)
                })
        
        return pd.DataFrame(predictions)
        
    except Exception as e:
        print(f"  [Deep] Model prediction failed: {e}")
        # Return empty DataFrame - caller will use fallback
        return pd.DataFrame(columns=['pid', 'time', 'predicted_value'])


def export_embeddings(
    dynamic_df: pd.DataFrame,
    model_path: str,
    output_path: str,
    feature: str = "eGFR"
) -> None:
    """
    Export patient embeddings from trained Deep model.
    
    Parameters
    ----------
    dynamic_df : pd.DataFrame
        Input data
    model_path : str
        Path to trained model
    output_path : str
        Where to save embeddings
    feature : str
        Feature to extract embeddings for
    """
    try:
        import torch
        from traj_ps.backends.deep.model import GRUDModel
        
        model = torch.load(model_path)
        model.eval()
        
        feature_df = dynamic_df[dynamic_df['feature_name'] == feature].copy()
        H_map = {}
        
        for pid in feature_df['pid'].unique():
            patient_data = feature_df[feature_df['pid'] == pid].sort_values('time')
            values = patient_data['value'].values
            
            X = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
            
            with torch.no_grad():
                # Extract hidden states from model
                embeddings = model.get_embeddings(X.unsqueeze(0))
                H_map[str(pid)] = embeddings.squeeze().cpu().numpy()
        
        save_embeddings(H_map, output_path)
        print(f"  [Deep] Saved embeddings to {output_path}")
        
    except Exception as e:
        print(f"  [Deep] Failed to export embeddings: {e}")
