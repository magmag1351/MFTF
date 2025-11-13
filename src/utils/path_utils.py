# src/utils/path_utils.py
import os
import re
from datetime import datetime

def get_latest_log_path(base_dir="data/logs"):
    """最新の日付のモニタログファイルを取得"""
    if not os.path.exists(base_dir):
        return None

    pattern = re.compile(r"monitor_log_(\d{8})\.csv")
    latest_file = None
    latest_date = None

    for fname in os.listdir(base_dir):
        match = pattern.match(fname)
        if match:
            date_str = match.group(1)
            try:
                date = datetime.strptime(date_str, "%Y%m%d")
                if latest_date is None or date > latest_date:
                    latest_date = date
                    latest_file = fname
            except ValueError:
                continue

    if latest_file:
        return os.path.join(base_dir, latest_file)
    else:
        return None
