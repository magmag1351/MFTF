import csv, os
from datetime import datetime
import threading

class DataLogger:
    def __init__(self, save_dir="data/logs"):
        os.makedirs(save_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        self.file_path = os.path.join(save_dir, f"monitor_log_{date_str}.csv")
        self.lock = threading.Lock()

        # CSV初期化（ヘッダーがなければ作成）
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "fatigue_score",
                    "head_tilt",
                    "keyboard_count",
                    "mouse_clicks",
                    "confidence"
                ])

    def log(self, fatigue_score, head_tilt, keyboard_count, mouse_clicks, confidence):
        """1分ごとにログを安全に追記"""
        with self.lock:
            try:
                with open(self.file_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        fatigue_score,
                        head_tilt,
                        keyboard_count,
                        mouse_clicks,
                        confidence
                    ])
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as e:
                print(f"[ERROR] ログ書き込み失敗: {e}")
