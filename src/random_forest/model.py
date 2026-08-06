import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier


class SymmetricForest(ClassifierMixin, BaseEstimator):
    """
    Random Forest constrained to satisfy p(x) = 1 - p(-x).

    Symmetry is enforced twice:
      * train time  -> each match is duplicated as (-x, 1 - y)
      * inference   -> p_hat = (p(x) + 1 - p(-x)) / 2

    Augmentation happens inside fit(), so GridSearchCV never leaks a
    mirrored row from the training fold into the validation fold.
    """

    def __init__(self, n_estimators=500, criterion='log_loss', max_depth=None,
                 min_samples_leaf=50, min_samples_split=2, max_features='sqrt',
                 random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)

        # mirror the dataset: odd features flip sign, label flips with them
        X_aug = np.vstack([X, -X])
        y_aug = np.concatenate([y, 1 - y])

        self.forest_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            bootstrap=True,
            oob_score=False,          # meaningless under augmentation
            class_weight=None,        # perfectly balanced by construction
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X_aug, y_aug)

        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p_fwd = self.forest_.predict_proba(X)[:, 1]
        p_rev = self.forest_.predict_proba(-X)[:, 1]
        p = 0.5 * (p_fwd + (1.0 - p_rev))     # exact antisymmetry
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)