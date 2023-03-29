print("Importing Libraries")
import cv2
import customUtils
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
from statistics import mode

print("Libraries Imported")

print("Loading DataPoints")
data = pd.read_csv("Data\data.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print("DataPoints Loaded")

print("Compiling RandomForestCLassifier")
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X.values, y.values)
print("RandomForestCLassifier Compiled")

print("Building Hand Landmark Detector")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
print("Hand Landmark Detector built")

print("Initializing Webcam")
now, prev = 0, 0
cap = cv2.VideoCapture(0)
capture_frame_threshold = 60
transition_frame_threshold = 20
transition_flag = False
output_letters = []
empty_frames = 0
word = ""
display_flag = 0

print("Initializing Parameters")
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2,
) as hands:
    print("Turning your webcam on, don't worry.")
    print("You can press 'esc' anytime to quit.")
    while True:
        ret, frame = cap.read()
        h, w, c = frame.shape
        cordinates, left, right = [], [], []
        pred_class = ""
        if not ret:
            print("Ignoring Empty Frames")
            continue
        frame = cv2.flip(frame, 1)

        if not transition_flag:
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True

            if results.multi_hand_landmarks:
                empty_frames = 0
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

                if len(cordinates) > 0 and len(cordinates) < 127:
                    pred_class = clf.predict([cordinates])[0]
                    output_letters.append(pred_class)

            if len(cordinates) == 0:
                output_letters.append("")
                empty_frames += 1

        else:
            counter += 1
            if counter == transition_frame_threshold:
                print("Transition Mode")
                transition_flag = False
                continue

        now = time.time()
        fps = 1 // (now - prev)
        prev = now
        frame = cv2.putText(
            frame,
            f"FPS = {fps}, Letter = {pred_class}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
        )

        if len(output_letters) == capture_frame_threshold:
            mode_letter = mode(output_letters)
            display_flag = 10
            word += mode_letter
            output_letters = []
            if mode_letter != "":
                transition_flag = True
                counter = 0

        if display_flag:
            display_flag = customUtils.displayMultiFrames(
                frame,
                f"Output Letter = {mode_letter}",
                (w // 2 - 50),
                (h // 2),
                display_flag,
            )

        if empty_frames == capture_frame_threshold:
            output = customUtils.autocorrect(word)
            print(output)
            word = ""

        cv2.imshow("Frame", frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
