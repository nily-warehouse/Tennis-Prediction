import os
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = _THIS_DIR / "kaggle-atp"

KAGGLE_DATASET_ID = "dissfya/atp-tennis-2000-2023daily-pull"

def pull_data() -> Path:

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Must be set BEFORE importing kagglehub
    os.environ["KAGGLEHUB_CACHE"] = str(DATA_DIR)
    import kagglehub

    dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))

    csv_file = next(dataset_path.glob("*.csv"), None)
    if csv_file is None:
        raise FileNotFoundError(
            f"No CSV found in downloaded dataset at {dataset_path}"
        )

    return csv_file

if __name__ == "__main__":
    print(pull_data())
    # must be -> '../src/data/kaggle-atp/datasets/dissfya/atp-tennis-2000-2023daily-pull/versions/1146/atp_tennis.csv'