import time, threading, random
from src.data_manager import DataLogger
from src.gui import FatigueGUI
from src.input_monitor import InputMonitor
import tkinter as tk

def collect_data(input_monitor):
    """1分ごとに入力数を集計し、疲労スコアを計算"""
    keyboard_count, mouse_clicks = input_monitor.get_counts_and_reset()

    # 仮の疲労スコア（ランダムに加味）
    fatigue_score = 40 + (keyboard_count + mouse_clicks) * 0.2 + random.uniform(-2, 2)
    head_tilt = random.uniform(5, 15)
    confidence = random.uniform(0.85, 0.95)

    return {
        "fatigue_score": round(fatigue_score, 2),
        "head_tilt": round(head_tilt, 2),
        "keyboard_count": keyboard_count,
        "mouse_clicks": mouse_clicks,
        "confidence": round(confidence, 2)
    }

def monitoring_loop(logger, input_monitor):
    """1分ごとのデータ収集・保存ループ"""
    while True:
        data = collect_data(input_monitor)
        logger.log(**data)
        print(f"[INFO] データ保存完了: {data}")
        time.sleep(60)

if __name__ == "__main__":
    logger = DataLogger()
    input_monitor = InputMonitor()
    input_monitor.start()  # 監視開始

    t = threading.Thread(target=monitoring_loop, args=(logger, input_monitor), daemon=True)
    t.start()

    root = tk.Tk()
    gui = FatigueGUI(root, logger.file_path)
    root.mainloop()

    # GUI終了後のクリーンアップ
    input_monitor.stop()
