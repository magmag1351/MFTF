# src/main.py
import time, threading, random
import tkinter as tk
from src.data_manager import DataLogger
from src.gui import FatigueGUI
from src.input_monitor import InputMonitor
from src.face_monitor import FaceMonitor
from src.fatigue_estimator import FatigueEstimator


def collect_data(input_monitor, face_monitor, estimator):
    """1分ごとに入力・顔データを集計して疲労スコアを算出"""
    keyboard_count, mouse_clicks = input_monitor.get_counts_and_reset()
    face_fatigue = face_monitor.get_fatigue_score()

    # 顔データが取得できない場合
    face_detected = face_fatigue is not None
    head_tilt = random.uniform(5, 15)  # 仮のデータ
    confidence = random.uniform(0.85, 0.95)

    # 疲労スコアを推定モデルから算出
    fatigue_score = estimator.estimate(
        head_tilt=head_tilt,
        keyboard_count=keyboard_count,
        mouse_clicks=mouse_clicks,
        face_detected=face_detected
    )

    return {
        "fatigue_score": round(fatigue_score, 2),
        "head_tilt": round(head_tilt, 2),
        "keyboard_count": keyboard_count,
        "mouse_clicks": mouse_clicks,
        "face_detected": face_detected,
        "confidence": round(confidence, 2)
    }


def monitoring_loop(logger, input_monitor, face_monitor):
    """1分ごとのデータ収集・保存ループ"""
    estimator = FatigueEstimator()

    while True:
        data = collect_data(input_monitor, face_monitor, estimator)
        logger.log(**data)
        print(f"[INFO] データ保存完了: {data}")
        time.sleep(60)


if __name__ == "__main__":
    logger = DataLogger()
    input_monitor = InputMonitor()
    face_monitor = FaceMonitor()

    input_monitor.start()

    # データ収集スレッド開始
    t = threading.Thread(target=monitoring_loop, args=(logger, input_monitor, face_monitor), daemon=True)
    t.start()

    # GUI起動
    root = tk.Tk()
    gui = FatigueGUI(root, logger.file_path)
    try:
        root.mainloop()
    finally:
        # 終了処理
        input_monitor.stop()
        face_monitor.release()
        print("[INFO] 監視を終了しました。")
