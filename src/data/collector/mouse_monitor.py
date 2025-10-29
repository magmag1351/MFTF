from pynput import mouse

class MouseMonitor:
    def __init__(self):
        self.click_count = 0
        listener = mouse.Listener(on_click=self.on_click)
        listener.daemon = True
        listener.start()

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.click_count += 1

    def get_activity_count(self):
        count = self.click_count
        self.click_count = 0
        return count
