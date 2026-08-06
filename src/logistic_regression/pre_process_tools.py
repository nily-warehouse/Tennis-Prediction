import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'Elo_diff',
    'Elo_mean',
    'N_min',
    'elo_surface_diff',
    'elo_effective_diff',
    'spec_diff',
    'surface_exp_min',
    'log_pts_diff',
    'log_rank_diff',
    # interactions (odd x even -> odd)
    'elo_x_rel',
    'elo_x_lowRel',
    'elo_x_clay',
    'elo_x_grass',
    'elo_x_bo5',
    'elo_x_field',
]


def impute_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Get rid of NaN instances."""
    df = df.copy()
    # its ok because we know pts NANs are just before 2003
    return df.dropna()


def drop_first_dummy(df: pd.DataFrame) -> pd.DataFrame:
    """Drop one of dummies,
    Because always one of them is a linear combination of others.
    """
    df = df.copy()
    # category with more instances 
    return df.drop(columns=['Surface_Hard'])


def split_data(df: pd.DataFrame, y_name, features=FEATURES) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split train/test and X/y data
    
    Returns:
        X_train, X_test, y_train, y_test
    """

    X = df[features].values
    y = df[y_name].values

    return train_test_split(
        X, y,
        test_size=0.2,
        shuffle=False # critical for series data
    )


def scale_features(X_train: np.ndarray, X_test: np.ndarray):
    """Scale odd features without centering, to preserve swap-symmetry."""

    # with_mean=False -> pure division by std, keeps f(-x) = -f(x)
    scaler = StandardScaler(with_mean=False)

    X_train_scaled = scaler.fit_transform(X_train)   # fit on train only
    X_test_scaled = scaler.transform(X_test)         # reuse train statistics

    return X_train_scaled, X_test_scaled


def add_symmetric_features(df, k=30):
    """Add odd (antisymmetric) interaction features: odd x even = odd."""
    d = df.copy()

    # even (invariant) moderators, scaled to [0, 1)
    rel = d["N_min"] / (d["N_min"] + k)

    # odd x even -> still odd
    d["elo_x_rel"]     = d["Elo_diff"] * rel
    d["elo_x_lowRel"]  = d["Elo_diff"] * (1 - rel)
    d["elo_x_clay"]    = d["Elo_diff"] * d["Surface_Clay"]
    d["elo_x_grass"]   = d["Elo_diff"] * d["Surface_Grass"]
    d["elo_x_bo5"]     = d["Elo_diff"] * d["Best_of_5"]
    d["elo_x_field"]   = d["Elo_diff"] * (d["Elo_mean"] - 1500) / 100

    return d