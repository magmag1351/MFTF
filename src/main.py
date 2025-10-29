from src.data.collector.keyboard_monitor import KeyboardMonitor
from src.data.collector.mouse_monitor import MouseMonitor
from src.data.collector.webcam_monitor import WebcamMonitor
from src.data.preprocessor.feature_extractor import FeatureExtractor
from src.model.fatigue_regressor import FatigueRegressor
from src.interface.notifier import Notifier
from src.utils.logger import setup_logger
import time

logger = setup_logger()

def main():
    logger.info("🔧 Fatigue Monitor 起動中...")

    keyboard = KeyboardMonitor()
    mouse = MouseMonitor()
    webcam = WebcamMonitor()
    feature_extractor = FeatureExtractor()
    model = FatigueRegressor()
    notifier = Notifier()

    try:
        while True:
            kb_count = keyboard.get_activity_count()
            mouse_count = mouse.get_activity_count()
            face_score = webcam.get_fatigue_indicator()

            features = feature_extractor.extract(kb_count, mouse_count, face_score)
            fatigue = model.predict(features)

            logger.info(f"📊 Current fatigue score: {fatigue:.2f}")

            if fatigue > 70:
                notifier.notify_rest(fatigue)

            time.sleep(60)  # 1分間隔で更新

    except KeyboardInterrupt:
        logger.info("🛑 モニタリングを終了しました。")

if __name__ == "__main__":
    main()
