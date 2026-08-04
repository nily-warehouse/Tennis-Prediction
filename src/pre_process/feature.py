import numpy  as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def add_log_pts_diff(df:pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()

    # monotone, symmetric, scale-invariant, sign carries direction
    df_['log_pts_diff'] = (
        np.log1p(df_['Pts_1']) - np.log1p(df_['Pts_2'])
    )

    return df_

def add_log_rank_diff(df:pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()

    # Rank is ordinal, so its diff is already scale-free by nature
    df_['log_rank_diff'] = (
        np.log1p(df_['Rank_1']) - np.log1p(df_['Rank_2'])
    )

    return df_

def encode_surface_types(df:pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()

    onehot = OneHotEncoder(sparse_output=False)
    df_['Surface'] = df_['Surface'].replace({'Carpet': 'Grass'})
    onehot_encoded = onehot.fit_transform(df_[['Surface']])

    surface_df = pd.DataFrame(
        onehot_encoded,
        # Surface_Clay, Surface_Grass, Surface_Hard
        columns=onehot.get_feature_names_out(['Surface']),
        index=df_.index,
    )
    df_ = pd.concat([df_, surface_df], axis=1)

    return df_

def encode_best_of(df:pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()
    df_['Best_of_5'] = (df_['Best of'] == 5).astype('int8')
    return df_