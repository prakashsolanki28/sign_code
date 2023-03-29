import cv2
import mediapipe as mp
import pandas as pd
import customUtils


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
cordinate_list = []
counter = {}

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2,
) as hands:

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()

        h, w, c = frame.shape

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        frame = cv2.flip(frame, 1)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_bgr.flags.writeable = False
        results = hands.process(frame_bgr)
        frame_bgr.flags.writeable = True
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            left = []
            right = []
            for handResult, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                x_min, y_min, z_min, x_max, y_max, z_max = customUtils.findMinMax(
                    handResult.landmark, w, h
                )
                if handedness.classification[0].label == "Left":
                    for landmark in handResult.landmark:
                        left.append(landmark.x - x_min)
                        left.append(landmark.y - y_min)
                        left.append(landmark.z - z_min)
                else:
                    for landmark in handResult.landmark:
                        right.append(landmark.x - x_min)
                        right.append(landmark.y - y_min)
                        right.append(landmark.z - z_min)
                cv2.rectangle(
                    frame,
                    (int(x_min * w), int(y_min * h)),
                    (int(x_max * w), int(y_max * h)),
                    (0, 255, 0),
                    1,
                )
                mp_drawing.draw_landmarks(
                    frame,
                    handResult,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )
            if len(left) == 0:
                left = [0] * 63
            if len(right) == 0:
                right = [0] * 63
            cordinates = left + right

        cv2.imshow("Input", frame)
        c = cv2.waitKey(1)

        try:
            if c == 27:
                break
            if c >= 97 and c <= 122 and len(cordinates) > 0 and len(cordinates) < 127:
                if c in counter:
                    counter[c] += 1
                else:
                    counter[c] = 1
                cordinates.append(chr(c))
                print(f"You pressed {chr(c)} {counter[c]} times!")
                cordinate_list.append(cordinates)
        except:
            print(
                "SORRY COULDNT LOG COORDINATES! IDK WHY SO ASK CJ AS HE IS SMARTER THAN YOU!"
            )

    df = pd.DataFrame.from_records(cordinate_list)
    df.to_csv("./Data/te.csv", index=False)
    cap.release()
    cv2.destroyAllWindows()
