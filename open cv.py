import cv2
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np

# Start webcam feed
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Setup for controlling audio
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax = volume.GetVolumeRange()[:2]
volBar = 400
volPer = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process hand detection
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handLandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handLandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            # Draw landmarks and connections
            mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)

    if lmList:
        x1, y1 = lmList[4][0], lmList[4][1] # Thumb tip
        x2, y2 = lmList[8][0], lmList[8][1] # Index finger tip

        # Draw circles on thumb and index finger
        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)

        # Draw line between thumb and index finger
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Calculate the length between the thumb and index finger
        length = hypot(x2 - x1, y2 - y1)

        # Interpolate volume based on the length
        vol = np.interp(length, [30, 350], [volMin, volMax])
        volBar = np.interp(length, [30, 350], [400, 150])
        volPer = np.interp(length, [30, 350], [0, 100])

        # Set the system volume
        volume.SetMasterVolumeLevel(vol, None)

        # Draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, f"{int(volPer)}%", (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # Display the image
    cv2.imshow("Image", img)

    # Break loop if spacebar is pressed
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
