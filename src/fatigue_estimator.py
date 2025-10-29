# src/fatigue_estimator.py
import numpy as np

class FatigueEstimator:
    def __init__(self):
        """
        疲労スコア推定モデル（初期版: 簡易線形モデル）
        ※後に学習済みモデルに置き換え可能
        """
        self.weights = {
            "head_tilt": 0.4,        # 首の傾きが大きいほど疲労上昇
            "keyboard_count": -0.2,  # 入力量が多いほど集中している可能性
            "mouse_clicks": -0.1,    # 活動量が多いほど集中
            "face_detected": -0.3,   # 顔が検出されないと疲労とみなす
        }

    def estimate(self, head_tilt, keyboard_count, mouse_clicks, face_detected):
        """与えられた特徴量から疲労スコア(0〜100)を推定"""
        w = self.weights

        score = (
            w["head_tilt"] * min(head_tilt / 45, 1.0) +  # 傾きを正規化
            w["keyboard_count"] * min(keyboard_count / 50, 1.0) +
            w["mouse_clicks"] * min(mouse_clicks / 50, 1.0) +
            w["face_detected"] * (0 if face_detected else 1)
        )

        fatigue_score = np.clip(50 + score * 50, 0, 100)
        return round(float(fatigue_score), 2)
