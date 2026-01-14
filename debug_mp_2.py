import sys
try:
    import mediapipe
    print("MP Imported")
    print(dir(mediapipe))
    import mediapipe.python.solutions
    print("Solutions imported explicitly")
except Exception as e:
    print(e)
