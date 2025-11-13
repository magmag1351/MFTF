# src/fatigue_estimator.py
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from src.utils.path_utils import get_latest_log_path

MODEL_PATH = "src/fatigue_model.pkl"
DATA_PATH = get_latest_log_path()


class FatigueEstimator:
    def __init__(self):
        self.model = None
        self.last_trained_count = 0  # 最後に学習したデータ数を記録
        self.load_model()

    def load_model(self):
        """保存済みモデルを読み込む"""
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            print(f"[INFO] モデルをロードしました: {MODEL_PATH}")
        else:
            print("[WARN] モデルが見つかりませんでした。predict()は無効です。")
            self.model = None

    def estimate(self, keyboard_count, mouse_clicks, head_tilt, confidence):
        """現在のデータから即時スコアを推定"""
        # 簡易スコア（学習とは独立）
        fatigue_score = 0.3 * keyboard_count + 0.1 * mouse_clicks + 2 * (head_tilt / 10) + (1 - confidence) * 50
        
        # ✅ 修正点: 戻り値を辞書にする
        return {
            "fatigue_score": round(fatigue_score, 2),
            "head_tilt": round(head_tilt, 2)
        }

    def predict(self, keyboard_count, mouse_clicks, head_tilt, confidence):
        """学習モデルによる次の1分の予測"""
        if self.model is None:
            print("[WARN] モデルが読み込まれていません。predictはスキップされます。")
            return None

        X = pd.DataFrame([[keyboard_count, mouse_clicks, head_tilt, confidence]],
                         columns=["keyboard_count", "mouse_clicks", "head_tilt", "confidence"])
        try:
            fatigue_score = self.model.predict(X)[0]
            return round(fatigue_score, 2)
        except Exception as e:
            print(f"[ERROR] 予測中に問題が発生: {e}")
            return None

    def train_if_needed(self):
        """データ件数が100件以上増えたら自動で再学習"""
        if not os.path.exists(DATA_PATH):
            return

        df = pd.read_csv(DATA_PATH)
        n_records = len(df)

        # 学習条件: データが100件以上、かつ前回から50件以上増加
        if n_records >= 100 and n_records - self.last_trained_count >= 50:
            print(f"[INFO] データが {n_records} 件に達しました。モデルを再学習します...")
            self.train(df)
            self.last_trained_count = n_records

    def train(self, df):
        """線形回帰モデルの再学習"""
        try:
            X = df[["keyboard_count", "mouse_clicks", "head_tilt", "confidence"]]
            y = df["fatigue_score"]

            model = LinearRegression()
            model.fit(X, y)
            joblib.dump(model, MODEL_PATH)

            self.model = model
            print(f"[INFO] モデルを再学習し、保存しました: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] 再学習中にエラー: {e}")
