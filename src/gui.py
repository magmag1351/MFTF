import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import os

class FatigueGUI:
    def __init__(self, root, csv_path):
        self.root = root
        self.csv_path = csv_path

        self.root.title("Fatigue Monitor Viewer")
        self.root.geometry("800x500")

        self.fig, self.ax = plt.subplots(figsize=(7, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.label = tk.Label(root, text="Loading...", font=("Arial", 14))
        self.label.pack(pady=10)

        self.update_graph()

    def update_graph(self):
        """CSV読み込み → グラフ更新（1分ごと）"""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            if not df.empty:
                self.ax.clear()
                self.ax.plot(df["timestamp"], df["fatigue_score"], label="Fatigue")
                self.ax.plot(df["timestamp"], df["head_tilt"], label="Head Tilt (deg)")
                self.ax.legend()
                self.ax.set_xlabel("Time")
                self.ax.set_ylabel("Value")
                self.fig.autofmt_xdate(rotation=45)
                self.canvas.draw()

                # 最新スコア表示
                last_score = df["fatigue_score"].iloc[-1]
                self.label.config(text=f"最新の疲労スコア: {last_score:.2f}")

        # 60秒ごとに再更新
        self.root.after(60000, self.update_graph)
