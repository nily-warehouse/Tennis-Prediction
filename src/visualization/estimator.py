import numpy as np
import pandas as pd
from models import predict as pred


def extract_data(player1, player2, meta_data):
    # playerX format:
    # 0 : name
    # 1 : generalElo
    # 2 : surfaceElo
    # 3 : pts
    # 4 : rank
    # 5 : matches
    # 6 : surfaceMatches
    # 7 : effectiveElo
    # 8 : spec

    # meta_data format:
    # 0 : surface
    # 1 : bestOf

    surface = meta_data[0]
    best_of = int(meta_data[1])

    data = [
        player1[1] - player2[1],          # Elo_diff
        (player1[1] + player2[1]) / 2,    # Elo_mean
        min(player1[5], player2[5]),      # N_min 
        player1[2] - player2[2],          # elo_surface_diff 
        player1[7] - player2[7],          # elo_effective_diff 
        player1[8] - player2[8],          # spec_diff 
        min(player1[6], player2[6]),      # surface_exp_min 
        log_diff(player1[3], player2[3]), # log_pts_diff 
        log_diff(player1[4], player2[4]), # log_rank_diff 
        1 if surface == 'Clay' else 0,    # Surface_Clay 
        1 if surface == 'Grass' else 0,   # Surface_Grass
        1 if surface == 'Hard' else 0,    # Surface_Hard
        1 if best_of == 5 else 0,         # Best_of_5
    ]

    return data

def log_diff(x, y):
    return np.log1p(x) - np.log1p(y)


def _to_native(value):
    if isinstance(value, dict):
        return {key: _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [_to_native(item) for item in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return _to_native(value.to_dict())
    if isinstance(value, pd.Index):
        return [_to_native(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_to_native(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _to_native(value.item())
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return value


def _estimate(player1, player2, meta_data, model):
    return pred([extract_data(player1, player2, meta_data)], model)


def estimate(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("Request payload must be a JSON object")

    required = ("player1", "player2", "meta_data", "model")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    probability = _estimate(
        payload["player1"],
        payload["player2"],
        payload["meta_data"],
        payload["model"],
    )
    return _to_native({"probability": probability})
