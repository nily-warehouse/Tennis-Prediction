import sys
from pathlib import Path
from datetime import datetime

# data exploration libraries
import numpy  as np
import pandas as pd

# pipline tools
sys.path.append(str(Path(__file__).parent.parent))
from data import download
from pre_process import (
    clean, 
    name_fix, 
    elo, 
    feature
)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "pre_processed"

def import_data() -> pd.DataFrame:
    return pd.read_csv(download.pull_data())

def clean_data(df:pd.DataFrame):

    df = clean.sort_data(df)
    df = clean.separate_year_column(df)

    df = clean.omit_unnecessary_columns(df)
    df = clean.handle_missings(df)

    df = name_fix.normalize_name_spaces(df)

    df = clean.add_is_winner_column(df)

    return df

def generate_Elo(df:pd.DataFrame):

    df = elo.add_general_Elo(df)
    df = elo.add_surface_Elo(df)

    return df

def remove_players_from_data(df:pd.DataFrame):

    df = feature. add_log_pts_diff(df)
    df = feature.add_log_rank_diff(df)

    df = feature.encode_best_of(df)
    df = feature.encode_surface_types(df)

    return df

def finalize_data(df:pd.DataFrame):
    return feature.export_final_data(df)

def save_data(df_processed: pd.DataFrame, elo_prob_index: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    features_path = PROCESSED_DIR / "features.parquet"
    baseline_path = PROCESSED_DIR / "elo_baseline.parquet"

    df_processed.to_parquet(
        features_path, engine="pyarrow", compression="snappy", index=False
    )

    elo_prob_index.to_parquet(
        baseline_path, engine="pyarrow", compression="snappy", index=False
    )

    for path in (features_path, baseline_path):
        print(f"path: {path} \n size: ({path.stat().st_size / 1e6:.2f} MB)")

def pre_process_pipline():

    # import
    print("🗿Importing dataset ...")
    df = import_data()
    print("✅Data imported!")

    # process
    print("🗿Pre-Processing started ...")
    df = clean_data(df)
    df = generate_Elo(df)
    df = remove_players_from_data(df)
    print("✅Data pre-processed!")

    # final result
    df_processed, elo_prob_index = finalize_data(df)

    # sample
    print()
    print("Processed data sample: (5-row)")
    print(df_processed.head(5))
    print("Processed data shape", df_processed.shape)
    print()
    print("elo prob sample: (5-row)")
    print(elo_prob_index.head(5))
    print("elo prob shape", elo_prob_index.shape)
    print()

    # save
    print("🗿Saving ...")
    save_data(df_processed, elo_prob_index)
    print("💾Saved!")

if __name__ == '__main__':
    pre_process_pipline()