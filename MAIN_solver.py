import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3

# === Setup TTS ===
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# === Setup MediaPipe ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# === Constants ===
finger_tips = [8, 12, 16, 20]
expression = ""
stable_gesture = None
stable_count = 0
STABLE_THRESHOLD = 10
last_input_time = 0
COOLDOWN_PERIOD = 1.5  # seconds
last_token_type = None

# === Webcam ===
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = None
    current_time = time.time()
    total_fingers = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            fingers_up = []

            # Thumb
            thumb_angle = calculate_angle(lm_list[2], lm_list[3], lm_list[4])
            thumb_tip, wrist, thumb_ip = lm_list[4], lm_list[0], lm_list[3]
            if thumb_angle > 160 and abs(thumb_tip[0] - wrist[0]) > 40 and thumb_tip[1] < thumb_ip[1]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

            # Other fingers
            for tip_id in finger_tips:
                fingers_up.append(1 if lm_list[tip_id][1] < lm_list[tip_id - 2][1] else 0)

            total_fingers += sum(fingers_up)

            # === Custom gesture logic for operators ===
            if fingers_up == [1, 0, 0, 0, 0]:
                gesture = '+'
            elif fingers_up == [1, 1, 0, 0, 0]:
                gesture = '-'
            elif fingers_up == [1, 1, 1, 0, 0]:
                gesture = '*'
            elif fingers_up == [1, 1, 1, 1, 0]:
                gesture = '/'
            elif fingers_up == [0, 0, 0, 0, 0]:
                gesture = '='
            elif fingers_up == [1, 0, 0, 0, 1]:
                gesture = 'C'

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # === Finger count-based digits ===
        if gesture is None:
            if total_fingers == 1:
                gesture = '1'
            elif total_fingers == 2:
                gesture = '2'
            elif total_fingers == 3:
                gesture = '3'
            elif total_fingers == 4:
                gesture = '4'
            elif total_fingers == 5:
                gesture = '5'
            elif total_fingers == 6:
                gesture = '6'
            elif total_fingers == 7:
                gesture = '7'
            elif total_fingers == 8:
                gesture = '8'
            elif total_fingers == 9:
                gesture = '9'

    # === Debounce + Cooldown ===
    if gesture == stable_gesture:
        stable_count += 1
    else:
        stable_gesture = gesture
        stable_count = 1

    if (stable_count >= STABLE_THRESHOLD and
        gesture is not None and
        (current_time - last_input_time) > COOLDOWN_PERIOD):

        last_input_time = current_time

        if gesture == '=':
            try:
                result = str(eval(expression))
                expression += ' = ' + result
                engine.say(f"equals {result}")
            except:
                expression += ' = ERR'
                engine.say("error")
            last_token_type = None

        elif gesture == 'C':
            expression = ''
            engine.say("cleared")
            last_token_type = None

        elif gesture in ['+', '-', '*', '/']:
            expression += f' {gesture} '
            engine.say(gesture)
            last_token_type = 'operator'

        elif gesture.isdigit():
            if last_token_type == 'digit':
                tokens = expression.strip().split(' ')
                if tokens and tokens[-1].isdigit():
                    tokens[-1] += gesture
                    expression = ' '.join(tokens)
                else:
                    expression += gesture
            else:
                expression += gesture
            engine.say(gesture)
            last_token_type = 'digit'

        engine.runAndWait()
        stable_count = 0

    # === UI Overlay ===
    cv2.putText(frame, f'Expr: {expression}', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, 'Press Q to quit', (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Math Solver (Two Hands)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
