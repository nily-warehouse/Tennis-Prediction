import unicodedata
import pandas as pd


def get_unique_names(df: pd.DataFrame) -> pd.Series:
    """Find unique names with their counts from match data."""
    return pd.concat([df.Player_1, df.Player_2]).value_counts()

def get_rare_names(df: pd.DataFrame, threshold=20) -> pd.Series:
    """Find players with lower experience then given threshold."""
    return get_unique_names(df)[get_unique_names(df) <= threshold]


def normalize_name_(s:str) -> str:
    """Fix space issues in a name."""
    s = s.fillna("").astype(str)

    # strip diacritics: Čilić -> Cilic
    s = s.map(lambda x: "".join(
        c for c in unicodedata.normalize("NFKD", x)
        if not unicodedata.combining(c)
    ))

    # inner space issues
    s = (s.str.replace(r"\s+", " ", regex=True) # collapse inner spaces
          .str.replace(r"\s*\.\s*", ".", regex=True) # "N ." -> "N."
          .str.strip())

    return s

def normalize_name_spaces(df:pd.DataFrame) -> pd.DataFrame:
    """Run normalizer for all names."""
    df_ = df.copy()

    for c in ["Player_1", "Player_2", "Winner"]:
        df_[c] = normalize_name_(df_[c])

    return df_