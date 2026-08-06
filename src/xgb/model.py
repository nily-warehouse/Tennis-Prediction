import numpy as np
import xgboost as xgb

def fit_sym(X, y, augment=True, **params):
    if augment:
        X, y = np.vstack([X, -X]), np.hstack([y, 1 - y])
    return xgb.XGBClassifier(eval_metric='logloss', **params).fit(X, y)

def predict_sym(model, X):
    """Exactly symmetric by construction: p(x) + p(-x) = 1"""
    return (model.predict_proba(X)[:, 1] + 1 - model.predict_proba(-X)[:, 1]) / 2