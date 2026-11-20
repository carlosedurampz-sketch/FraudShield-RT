"""
Isotonic-calibrated wrapper (sklearn >= 1.6 removed CalibratedClassifierCV(cv="prefit")).

Fits an IsotonicRegression on the base model's raw scores on the
calibration split, then applies it at scoring time. The wrapper exposes
the standard ``predict`` / ``predict_proba`` interface, so the FastAPI
app and joblib persistence work unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibratedModel:
    def __init__(self, base_model: Any = None, isotonic: Any = None) -> None:
        self.base_model = base_model
        self.isotonic = isotonic or IsotonicRegression(out_of_bounds="clip")

    def fit_calibration(
        self, X_cal: np.ndarray, y_cal: np.ndarray
    ) -> IsotonicCalibratedModel:
        raw = self.base_model.predict_proba(X_cal)[:, 1]
        self.isotonic.fit(raw, y_cal)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.isotonic.predict(raw)
        cal = np.clip(cal, 0.0, 1.0)
        return np.column_stack([1 - cal, cal])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        return getattr(self.base_model, "feature_importances_", None)
