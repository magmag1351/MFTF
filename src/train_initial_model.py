# src/train_initial_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

MODEL_PATH = "src/fatigue_model.pkl"
DATA_PATH = "data/logs/monitor_log_20251112.csv"  # DataLoggerで保存されるCSVを利用

def train_initial_model():
    if not os.path.exists(DATA_PATH):
        print("[ERROR] 学習データが存在しません。先にアプリを少し動かしてデータを生成してください。")
        return

    df = pd.read_csv(DATA_PATH)

    if len(df) < 10:
        print("[WARN] データが少なすぎます。最低でも10件以上の記録を推奨します。")
    
    # ===== 特徴量と目的変数 =====
    X = df[["keyboard_count", "mouse_clicks", "head_tilt", "confidence"]]
    y = df["fatigue_score"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] 初期線形モデルを保存しました: {MODEL_PATH}")
    print(f"[INFO] 学習データ件数: {len(df)}")

if __name__ == "__main__":
    train_initial_model()
