import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Tip landmarks for non-thumb fingers
finger_tips = [8, 12, 16, 20]

# Calculate angle between 3 points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            lm_list = []

            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            fingers_up = []

            # 🧠 Thumb logic (combined angle + distance + direction)
            thumb_angle = calculate_angle(lm_list[2], lm_list[3], lm_list[4])
            thumb_tip = lm_list[4]
            thumb_ip = lm_list[3]
            thumb_mcp = lm_list[2]
            thumb_cmc = lm_list[1]
            wrist = lm_list[0]

            # Conditions for thumb being UP
            if (
                thumb_angle > 160 and
                abs(thumb_tip[0] - wrist[0]) > 40 and
                thumb_tip[1] < thumb_ip[1]
            ):
                fingers_up.append(1)
            else:
                fingers_up.append(0)

            # ✌️ Check other four fingers
            for tip_id in finger_tips:
                if lm_list[tip_id][1] < lm_list[tip_id - 2][1]:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            total_fingers = sum(fingers_up)

            # Show finger count
            cv2.putText(frame, f'Fingers: {total_fingers}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Draw hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
