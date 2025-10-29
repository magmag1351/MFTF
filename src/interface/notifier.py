import tkinter as tk
from threading import Thread

class Notifier:
    def notify_rest(self, fatigue_score):
        Thread(target=self._show_popup, args=(fatigue_score,)).start()

    def _show_popup(self, fatigue_score):
        root = tk.Tk()
        root.title("休憩のおすすめ")
        label = tk.Label(
            root, text=f"疲労度 {fatigue_score:.1f}%。少し休憩しましょう☕", padx=20, pady=20
        )
        label.pack()
        button = tk.Button(root, text="OK", command=root.destroy)
        button.pack(pady=10)
        root.mainloop()
