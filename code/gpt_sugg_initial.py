# Fit linear mixed-effects model to extract subject-specific trajectories
from statsmodels.regression.mixed_linear_model import MixedLM
import pandas as pd
import numpy as np


def fit_trajectory_model(df, id_col, time_col, value_col):
    df['time_sq'] = df[time_col] ** 2
    model = MixedLM(df[value_col], df[[time_col, 'time_sq']], groups=df[id_col])
    result = model.fit()
    df['predicted'] = result.predict()
    return result


# Cluster random effect coefficients to classify trajectory types
from sklearn.mixture import GaussianMixture


def classify_trajectories(mixed_model_result, n_groups=5):
    coefs = mixed_model_result.random_effects
    coef_matrix = np.stack([list(coefs[i].values()) for i in coefs])
    gmm = GaussianMixture(n_components=n_groups, random_state=42).fit(coef_matrix)
    labels = gmm.predict(coef_matrix)
    return pd.DataFrame({'patient_id': list(coefs.keys()), 'trajectory_class': labels})


def prepare_data(time_varying_covs, traj_classes, treatments):
    df = time_varying_covs.merge(traj_classes, on='patient_id')
    df = df.merge(treatments, on=['patient_id', 'time'])
    return df


from sklearn.linear_model import LogisticRegression


def time_dependent_ps(df, time_col='time', treat_col='treatment', covariates=None):
    results = []
    for t in sorted(df[time_col].unique()):
        at_risk = df[df[time_col] == t]
        X = at_risk[covariates]
        y = at_risk[treat_col]
        model = LogisticRegression().fit(X, y)
        at_risk = at_risk.copy()
        at_risk['ps'] = model.predict_proba(X)[:, 1]
        results.append(at_risk)
    return pd.concat(results)


# RNN
import torch.nn as nn


class TrajEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        _, h_n = self.rnn(x)
        return h_n[-1]  # shape: [batch_size, hidden_dim]


class TimeVaryingPSModel(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq):
        repr = self.encoder(x_seq)
        logits = self.linear(repr)
        return nn.sigmoid(logits)


def train_time_bin_models(data_loader_dict, model, optimizer, criterion, epochs=10):
    model.train()
    for t, loader in data_loader_dict.items():
        for _ in range(epochs):
            for x_seq, treatment in loader:
                optimizer.zero_grad()
                ps = model(x_seq).squeeze()
                loss = criterion(ps, treatment.float())
                loss.backward()
                optimizer.step()


if __name__ == '__main__':
    import patsy

    print(patsy.__version__)
    from patsy import dmatrix
