# src/gui.py
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib

# フォント警告（CJK文字欠落）を抑制
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'Yu Gothic'  # Windows向け日本語フォント

class FatigueGUI:
    def __init__(self, root, csv_path):
        self.root = root
        self.csv_path = csv_path

        self.root.title("Fatigue Monitor Viewer")
        self.root.geometry("900x550")
        self.root.configure(bg="#f7f7f7")

        # ====== Matplotlib グラフエリア ======
        self.fig, self.ax = plt.subplots(figsize=(7.5, 3.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=10)

        # ====== ラベルエリア ======
        frame = tk.Frame(root, bg="#f7f7f7")
        frame.pack(pady=10)

        self.label_current = tk.Label(
            frame, text="現在の疲労スコア: --", font=("Arial", 13), bg="#f7f7f7"
        )
        self.label_current.grid(row=0, column=0, padx=20)

        self.label_predicted = tk.Label(
            frame, text="次の1分間予測: --", font=("Arial", 13), fg="blue", bg="#f7f7f7"
        )
        self.label_predicted.grid(row=0, column=1, padx=20)

        self.label_info = tk.Label(
            root, text="", font=("Arial", 11), fg="gray", bg="#f7f7f7"
        )
        self.label_info.pack()

        # ====== 初回描画 ======
        self.update_graph()

    def update_graph(self):
        """CSV読み込み → グラフ更新（1分ごと）"""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)

            if not df.empty:
                self.ax.clear()

                # ==== グラフ描画 ====
                self.ax.plot(
                    df["timestamp"],
                    df["fatigue_score"],
                    label="Fatigue (実測)",
                    color="orange",
                    marker="o",
                )

                # 予測スコアが存在する場合のみ描画
                if "predicted_score" in df.columns:
                    self.ax.plot(
                        df["timestamp"],
                        df["predicted_score"],
                        label="Predicted (予測)",
                        color="blue",
                        linestyle="--",
                    )

                # 頭部傾きデータも表示
                if "head_tilt" in df.columns:
                    self.ax.plot(
                        df["timestamp"],
                        df["head_tilt"],
                        label="Head Tilt (deg)",
                        color="green",
                        alpha=0.6,
                    )

                # グラフ装飾
                self.ax.legend()
                self.ax.set_xlabel("Time")
                self.ax.set_ylabel("Value")
                self.ax.set_title("Fatigue Monitoring Trend", fontsize=12)
                self.fig.autofmt_xdate(rotation=45)
                self.canvas.draw()

                # ==== 最新値表示 ====
                last_row = df.iloc[-1]

                # 実測スコア
                fatigue_score = last_row.get("fatigue_score", None)
                if pd.notna(fatigue_score):
                    self.label_current.config(
                        text=f"現在の疲労スコア: {fatigue_score:.2f}"
                    )
                else:
                    self.label_current.config(text="現在の疲労スコア: --")

                # 予測スコア
                predicted = last_row.get("predicted_score", None)
                if pd.notna(predicted):
                    self.label_predicted.config(
                        text=f"次の1分間予測: {predicted:.2f}"
                    )
                else:
                    self.label_predicted.config(text="次の1分間予測: --")

                # 更新時刻
                timestamp = last_row.get("timestamp", "")
                self.label_info.config(text=f"データ更新時刻: {timestamp}")

        else:
            self.label_info.config(text="ログファイルが見つかりません。")

        # 60秒ごとに再更新
        self.root.after(60000, self.update_graph)
