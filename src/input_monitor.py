from pynput import keyboard, mouse
import threading

class InputMonitor:
    def __init__(self):
        self.keyboard_count = 0
        self.mouse_clicks = 0
        self._lock = threading.Lock()
        self._running = False

    def on_key_press(self, key):
        with self._lock:
            self.keyboard_count += 1

    def on_click(self, x, y, button, pressed):
        if pressed:
            with self._lock:
                self.mouse_clicks += 1

    def start(self):
        """監視スレッドを開始"""
        self._running = True
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_press)
        self.mouse_listener = mouse.Listener(on_click=self.on_click)
        self.keyboard_listener.start()
        self.mouse_listener.start()

    def stop(self):
        """監視を停止"""
        self._running = False
        self.keyboard_listener.stop()
        self.mouse_listener.stop()

    def get_counts_and_reset(self):
        """現在のカウントを取得し、0にリセット"""
        with self._lock:
            k, m = self.keyboard_count, self.mouse_clicks
            self.keyboard_count = 0
            self.mouse_clicks = 0
            return k, m
