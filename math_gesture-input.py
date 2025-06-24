import cv2
import mediapipe as mp
import numpy as np
import time

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks (ignore thumb for now)
finger_tips = [8, 12, 16, 20]

# Calculate angle for thumb
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Hand detection + logic
cap = cv2.VideoCapture(0)
prev_finger_count = -1
last_added_time = time.time()
expression = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_time = time.time()

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            lm_list = []

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            fingers_up = []

            # Thumb logic
            thumb_angle = calculate_angle(lm_list[2], lm_list[3], lm_list[4])
            thumb_tip = lm_list[4]
            wrist = lm_list[0]
            thumb_ip = lm_list[3]

            if thumb_angle > 160 and abs(thumb_tip[0] - wrist[0]) > 40 and thumb_tip[1] < thumb_ip[1]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

            # Other fingers
            for tip_id in finger_tips:
                if lm_list[tip_id][1] < lm_list[tip_id - 2][1]:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            finger_count = sum(fingers_up)

            # Only add if number changed and hand is steady
            if finger_count != prev_finger_count:
                last_added_time = current_time
                prev_finger_count = finger_count

            if current_time - last_added_time > 1.0 and 0 < finger_count <= 5:
                expression += str(finger_count)
                print("Added digit:", finger_count)
                last_added_time = current_time  # Reset timer
                prev_finger_count = -1  # Reset detection

            # Draw hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay expression text
    cv2.putText(frame, f'Expression: {expression}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)

    cv2.imshow("Math Input", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
