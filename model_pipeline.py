import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class PyrolysisReactorPipeline(BaseEstimator, RegressorMixin):

  def __init__(self, solid_model=None, liquid_model=None):
    self.solid_model = solid_model
    self.liquid_model = liquid_model
    self.feature_names_in_ = None

  def predict(self, X):
    if isinstance(X, np.ndarray) and self.feature_names_in_ is not None:
      X = pd.DataFrame(X, columns=self.feature_names_in_)

    pred_solid = self.solid_model.predict(X)
    pred_liquid = self.liquid_model.predict(X)
    pred_gas = np.maximum(0.0, 100.00 - (pred_solid + pred_liquid))

    return {
        "solid": pred_solid,
        "liquid": pred_liquid,
        "gas": pred_gas,
        "summary_df": pd.DataFrame({
            "Solid (%)": pred_solid,
            "Liquid (%)": pred_liquid,
            "Gas (%)": pred_gas,
        }),
    }