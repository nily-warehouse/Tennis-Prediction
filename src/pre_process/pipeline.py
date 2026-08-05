"""ATP tennis pre-processing pipeline.

Download raw ATP match data, clean it, compute Elo ratings, engineer
features, and export the training-ready dataset to Parquet.

Steps:
    1. Import raw data from Kaggle.
    2. Clean columns, missing values, and player names.
    3. Compute general and surface-specific Elo ratings.
    4. Transform numeric and categorical features.
    5. Export features and the Elo baseline as Parquet.

Outputs:
    features.parquet: Final training features.
    elo_baseline.parquet: Elo-probability baseline index.
"""

import sys
from pathlib import Path
import pandas as pd

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
    """Download ATP-tennis data from Kaggle and import as Pandas DataFrame.

    Returns:
        Raw match data
    """
    return pd.read_csv(download.pull_data())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data and prepare it for feature engineering.

    Args:
        df: Raw match data
    
    Returns:
        Cleaned data
    """
    df = df.copy()

    # Initial processes on data
    df = clean.sort_data(df)
    df = clean.separate_year_column(df)
    df = clean.omit_unnecessary_columns(df)

    # Replace -1, 0 or empty cells with np.nan 
    df = clean.handle_missings(df)

    # Remove invalid spaces in names to avoid typo
    df = name_fix.normalize_name_spaces(df)

    # Replace winner name with a binary is-player-1-winner variable
    df = clean.add_is_winner_column(df)

    return df


def generate_elo(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Elo rating for both players.

    Args:
        df: Match data without Elo rating

    Returns:
        Match data with general and surface Elo.
    """
    df = df.copy()

    # Add Elo for each player based on all matches
    df = elo.add_general_Elo(df)

    # Add Elo for each player based on matches on a specific surface
    df = elo.add_surface_Elo(df)

    return df


def feature_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transform numeric features into log1p differences and encode categoricals.

    Args:
        df: cleaned data with Elo

    Returns:
        Transformed match data
    """
    df = df.copy()

    # Transform Pts & Rank data
    df = feature.add_log_pts_diff(df)
    df = feature.add_log_rank_diff(df)

    # Encode categorical features
    df = feature.encode_best_of(df)
    df = feature.encode_surface_types(df)

    return df


def finalize_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the final training features and the Elo-probability baseline.

    Args:
        df: pre-processed data

    Returns:
        tuple of (final features, elo_prob index)
    """
    df = df.copy()
    return feature.export_final_data(df)

def save_data(df_processed: pd.DataFrame, elo_prob_index: pd.DataFrame) -> None:
    """Save pre-processed data as Parquet form.

    Args:
        df_processed  : Final data
        elo_prob_index: elo_prob index
    """

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


def pre_process_pipeline() -> None:
    """Run the full pre-processing pipeline."""

    print("🗿Importing dataset ...")
    df = import_data()
    print("✅Data imported!")

    print("🗿Pre-Processing started ...")
    df = clean_data(df)
    df = generate_elo(df)
    df = feature_transform(df)
    print("✅Data pre-processed!")

    df_processed, elo_prob_index = finalize_data(df)

    print()
    print("Processed data sample: (5-row)")
    print(df_processed.head(5))
    print("Processed data shape", df_processed.shape)
    print()
    print("Elo prob sample: (5-row)")
    print(elo_prob_index.head(5))
    print("Elo prob shape", elo_prob_index.shape)
    print()

    print("🗿Saving ...")
    save_data(df_processed, elo_prob_index)
    print("💾Saved!")

if __name__ == '__main__':
    pre_process_pipeline()