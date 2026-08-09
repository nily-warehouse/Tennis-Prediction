from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add parent dir to path so sibling packages can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd

# Heavy/optional imports are needed only for type hints.
# With `from __future__ import annotations` they are never
# evaluated at runtime, so keeping them here avoids slow imports.
if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from random_forest.model import SymmetricForest
    from xgboost import XGBClassifier
    from tensorflow.python.keras import Sequential

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Anchor paths to this file, not the current working directory
BASE_DIR = Path(__file__).resolve().parent
MODELS_PATH = BASE_DIR.parent / "data" / "trained_models"
SCALERS_PATH = BASE_DIR.parent / "data" / "scaler"

RAW_FEATURES_NAME = [
    "Elo_diff", "Elo_mean", "N_min",
    "elo_surface_diff", "elo_effective_diff", "spec_diff",
    "surface_exp_min", "log_pts_diff", "log_rank_diff",
    "Surface_Clay", "Surface_Grass", "Surface_Hard", "Best_of_5",
]


# --- Generic loader ---

def _load_pickle(path: Path, description: str):
    """Load a pickled object and log progress."""
    logger.info("loading %s ...", description)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    logger.info("loaded successfully!")
    return obj


# --- Loaders ---

def logistic_reg() -> "LogisticRegression":
    return _load_pickle(MODELS_PATH / "logistic_model.pkl", "Logistic Regression model")


def logistic_reg_scaler() -> "StandardScaler":
    return _load_pickle(SCALERS_PATH / "logistic_reg_scaler.pkl", "Logistic Regression scaler")


def random_forest() -> "SymmetricForest":
    return _load_pickle(MODELS_PATH / "rf_model.pkl", "Random Forest model")


def xgboost() -> "XGBClassifier":
    return _load_pickle(MODELS_PATH / "xgb_model.pkl", "XGBoost model")


def dnn() -> "Sequential":
    return _load_pickle(MODELS_PATH / "dnn_model.pkl", "Deep Neural Network model")


def dnn_scaler() -> "StandardScaler":
    return _load_pickle(SCALERS_PATH / "dnn_scaler.pkl", "Deep Neural Network scaler")


# --- Predictions ---

def logistic_reg_prediction(data):
    # Imported lazily: the preprocessing package is only needed at predict time
    import logistic_regression.pre_process_tools as tools
    from logistic_regression.pre_process_tools import FEATURES as FEATURES_NAME

    model = logistic_reg()
    scaler = logistic_reg_scaler()

    # Prepare data for the logistic model
    data = pd.DataFrame(data)
    data.columns = RAW_FEATURES_NAME
    data = tools.drop_first_dummy(data)
    data = tools.add_symmetric_features(data)
    data = data[FEATURES_NAME]
    data = tools.scale_features(data, data, scaler=scaler)[0]

    return model.predict_proba(data)[0][1]


def random_forest_prediction(data):
    return random_forest().predict_proba(data)[0][1]


def xgboost_prediction(data):
    return xgboost().predict_proba(data)[0][1]


def dnn_prediction(data):
    import logistic_regression.pre_process_tools as tools

    model = dnn()
    scaler = dnn_scaler()

    data = tools.scale_features(data, data, scaler=scaler)[0]
    return model.predict(data)[0][0]


# --- Selection ---

_PREDICTORS = {
    "logistic_regression": logistic_reg_prediction,
    "random_forest": random_forest_prediction,
    "xgboost": xgboost_prediction,
    "dnn": dnn_prediction,
}


def predict(data, model: str):
    try:
        return _PREDICTORS[model](data)
    except KeyError:
        raise ValueError(
            f"Unknown model: {model!r}. Choose from {list(_PREDICTORS)}"
        )