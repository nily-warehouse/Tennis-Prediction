import numpy  as np
import pandas as pd


def sort_data(df:pd.DataFrame) -> pd.DataFrame:
    """Sort data with `Date` column."""
    df_ = df.copy()
    df_['Date'] = pd.to_datetime(df_['Date'], errors='coerce')
    return df_.sort_values('Date', ascending=True)


def separate_year_column(df:pd.DataFrame) -> pd.DataFrame:
    """Seprate `Year` for usages in exploartions."""
    df_ = df.copy()
    df_['Year'] = df_['Date'].dt.year
    return df_


def omit_unnecessary_columns(df:pd.DataFrame) -> pd.DataFrame:
    """Omit metadata features."""
    return df.drop(
        [
            'Tournament',
            'Date',
            'Series',
            'Court',
            'Round',
            'Odd_1',
            'Odd_2',
            'Score',
            'Year'
        ],
        axis=1
    )


def handle_missings(df:pd.DataFrame) -> pd.DataFrame:
    """Convert missings with NaN."""
    df_ = df.copy()

    for c in ["Rank_1", "Rank_2", "Pts_1", "Pts_2"]:
        df_.loc[df_[c] <= 0, c] = np.nan

    return df_.dropna(subset=['Rank_1', 'Rank_2'])


def add_is_winner_column(df:pd.DataFrame) -> pd.DataFrame:
    """Convert winner name to a is-player-1-winner form."""
    df_ = df.copy()
    df_["Winner"] = (df_.Winner == df_.Player_1).astype(int)
    return df_