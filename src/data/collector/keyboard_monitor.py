from pynput import keyboard

class KeyboardMonitor:
    def __init__(self):
        self.count = 0
        listener = keyboard.Listener(on_press=self.on_press)
        listener.daemon = True
        listener.start()

    def on_press(self, key):
        self.count += 1

    def get_activity_count(self):
        count = self.count
        self.count = 0
        return count
