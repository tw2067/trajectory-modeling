import numpy as np
from traj_ps.config import load_configs
from traj_ps.inference.export_embed import embedding_map_to_df

def test_load_configs_min():
    data, train = load_configs(None, None)
    assert "bin_width" in data and data["bin_width"] > 0

def test_embedding_export_min():
    H_map = {"p1": np.zeros((3,4)), "p2": np.ones((2,4))}
    df = embedding_map_to_df(H_map)
    assert set(["pid","t_idx","h1","h2","h3","h4"]).issubset(df.columns)
