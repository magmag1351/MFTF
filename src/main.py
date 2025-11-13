# src/main.py
import time
import threading
import random
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

    face_detected = face_fatigue is not None
    head_tilt = random.uniform(5, 15)  # 仮のデータ
    confidence = random.uniform(0.85, 0.95)

    current = estimator.estimate(
        keyboard_count, mouse_clicks, head_tilt, face_fatigue
    )

    predicted_score = estimator.predict(
        keyboard_count, mouse_clicks, head_tilt, confidence
    )

    data = {
        "fatigue_score": current["fatigue_score"],
        "keyboard_count": keyboard_count,
        "mouse_clicks": mouse_clicks,
        "face_detected": face_detected,
        "confidence": confidence,
        "predicted_score": predicted_score
    }

    return data


def monitoring_loop(logger, input_monitor, face_monitor, stop_event):
    """1分ごとのデータ収集・保存ループ（停止可能）"""
    estimator = FatigueEstimator()

    while not stop_event.is_set():  # ← フラグで停止監視
        data = collect_data(input_monitor, face_monitor, estimator)
        logger.log(**data)

        # データ件数が100件を超えたら自動再学習
        estimator.train_if_needed()

        print(f"[INFO] データ保存完了: {data}")

        # 1分間スリープ。ただし途中で停止要求が来たらすぐ抜ける
        for _ in range(60):
            if stop_event.is_set():
                break
            time.sleep(1)


if __name__ == "__main__":
    logger = DataLogger()
    input_monitor = InputMonitor()
    face_monitor = FaceMonitor()
    stop_event = threading.Event()  # ← 停止用フラグ

    input_monitor.start()

    # データ収集スレッド開始
    t = threading.Thread(
        target=monitoring_loop,
        args=(logger, input_monitor, face_monitor, stop_event),
        daemon=True
    )
    t.start()

    # GUI起動
    root = tk.Tk()
    gui = FatigueGUI(root, logger.file_path)

    def on_close():
        """GUIの×ボタンで安全に終了"""
        print("[INFO] 終了処理を開始します...")
        stop_event.set()       # 監視ループに停止信号
        input_monitor.stop()
        face_monitor.release()
        root.destroy()
        print("[INFO] 監視を終了しました。")

    root.protocol("WM_DELETE_WINDOW", on_close)  # GUI閉じるとon_close実行

    try:
        root.mainloop()
    except KeyboardInterrupt:
        # Ctrl+C対応
        on_close()
