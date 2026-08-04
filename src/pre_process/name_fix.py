import unicodedata
import pandas as pd

# --- get names ---

def get_unique_names(df:pd.DataFrame) -> pd.Series:
    return pd.concat([df.Player_1, df.Player_2]).value_counts()

def get_rare_names(df:pd.DataFrame) -> pd.Series:
    return get_unique_names(df)[get_unique_names(df) <= 20]

# --- util ---

def normalize_name_(s):
    s = s.fillna("").astype(str)
    # strip diacritics: Čilić -> Cilic
    s = s.map(lambda x: "".join(
        c for c in unicodedata.normalize("NFKD", x)
        if not unicodedata.combining(c)
    ))
    s = (s.str.replace(r"\s+", " ", regex=True) # collapse inner spaces
          .str.replace(r"\s*\.\s*", ".", regex=True) # "N ." -> "N."
          .str.strip())
    return s

# --- normalizer ---

def normalize_name_spaces(df:pd.DataFrame) -> pd.DataFrame:
    df_ = df.copy()
    for c in ["Player_1", "Player_2", "Winner"]:
        df_[c] = normalize_name_(df_[c])
    return df_