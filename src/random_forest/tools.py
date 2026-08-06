import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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
    'Surface_Hard',
    'Surface_Clay',
    'Surface_Grass',
    'Best_of_5',
]


def impute_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Get rid of NaN instances."""
    df = df.copy()
    # its ok because we know pts NANs are just before 2003
    return df.dropna()


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
