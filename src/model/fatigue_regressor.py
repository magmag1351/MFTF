import numpy as np
from sklearn.linear_model import LinearRegression

class FatigueRegressor:
    def __init__(self):
        # 初期状態では簡易モデル
        self.model = LinearRegression()
        self.model.coef_ = np.array([0.3, 0.3, -0.2])
        self.model.intercept_ = 50

    def predict(self, features):
        score = np.dot(features, self.model.coef_) + self.model.intercept_
        return np.clip(score, 0, 100)
