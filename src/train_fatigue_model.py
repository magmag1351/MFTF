# src/train_fatigue_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# データ読み込み
df = pd.read_csv("data/fatigue_log.csv")

# 特徴量と目的変数
X = df[["head_tilt", "keyboard_count", "mouse_clicks", "confidence"]]
y = df["fatigue_score"]

# 線形回帰モデルを学習
model = LinearRegression()
model.fit(X, y)

# モデルを保存
joblib.dump(model, "models/fatigue_model.pkl")
print("[INFO] モデルを保存しました: models/fatigue_model.pkl")
