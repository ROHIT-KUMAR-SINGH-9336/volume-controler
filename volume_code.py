import cv2
import mediapipe as mp
import math
import numpy as np
import time
import keyboard
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Audio setup (fixed ._iid_ and CLSCTX_ALL usage)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

TIP_IDS = [4, 8, 12, 16, 20]

def fingers_up(hand_landmarks):
    fingers = []
    # Thumb (horizontal)
    if hand_landmarks.landmark[TIP_IDS[0]].x < hand_landmarks.landmark[TIP_IDS[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers (vertical)
    for i in range(1, 5):
        if hand_landmarks.landmark[TIP_IDS[i]].y < hand_landmarks.landmark[TIP_IDS[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

cap = cv2.VideoCapture(0)

is_muted = False
previous_x = None
gesture_start_x = None
GESTURE_THRESHOLD = 80  # Swipe sensitivity

last_track_change_time = 0
track_cooldown = 1.5  # seconds cooldown for track change

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_volume = volume.GetMasterVolumeLevelScalar() * 100

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_status = fingers_up(hand_landmarks)

            x_thumb = int(hand_landmarks.landmark[4].x * frame.shape[1])
            y_thumb = int(hand_landmarks.landmark[4].y * frame.shape[0])
            x_index = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y_index = int(hand_landmarks.landmark[8].y * frame.shape[0])
            cx = (x_thumb + x_index) // 2

            # Fist = Mute
            if finger_status == [0, 0, 0, 0, 0] and not is_muted:
                volume.SetMute(1, None)
                is_muted = True
                print("🔇 Muted")
                gesture_start_x = None

            # Thumbs Up = Unmute
            elif finger_status == [1, 0, 0, 0, 0] and is_muted:
                volume.SetMute(0, None)
                is_muted = False
                print("🔊 Unmuted")
                gesture_start_x = None

            # Thumb + Index = Volume control
            elif finger_status == [1, 1, 0, 0, 0]:
                distance = math.hypot(x_index - x_thumb, y_index - y_thumb)
                vol_percent = np.interp(distance, [30, 200], [0, 100])
                vol_percent = np.clip(vol_percent, 0, 100)
                volume.SetMasterVolumeLevelScalar(vol_percent / 100.0, None)
                print(f"🔉 Volume: {int(vol_percent)}%")
                cv2.circle(frame, (cx, (y_thumb + y_index) // 2), 10, (0, 255, 0), cv2.FILLED)
                gesture_start_x = None

            # Open Palm (4 fingers up) = Swipe for track change
            elif finger_status == [0, 1, 1, 1, 1]:
                if gesture_start_x is None:
                    gesture_start_x = cx
                elif previous_x is not None:
                    delta_x = cx - previous_x
                    if abs(cx - gesture_start_x) > GESTURE_THRESHOLD:
                        now = time.time()
                        if now - last_track_change_time > track_cooldown:
                            if delta_x > GESTURE_THRESHOLD:
                                print("➡ Next Track")
                                keyboard.send('media next')
                            elif delta_x < -GESTURE_THRESHOLD:
                                print("⬅ Previous Track")
                                keyboard.send('media previous')
                            last_track_change_time = now
                            gesture_start_x = None
                cv2.putText(frame, "Swipe to Change Track", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                gesture_start_x = None

            previous_x = cx

    # Volume slider
    cv2.rectangle(frame, (50, 450), (450, 470), (255, 255, 255), 2)
    cv2.rectangle(frame, (50, 450), (int(50 + current_volume * 4), 470), (0, 255, 0), -1)
    cv2.putText(frame, f"{int(current_volume)}%", (460, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Mute status
    cv2.putText(frame, f'Status: {"Muted" if is_muted else "Unmuted"}',
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255) if is_muted else (0, 255, 0), 2)

    cv2.imshow("Gesture Audio Control (Windows)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
