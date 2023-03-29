from english_words import get_english_words_set
import nltk
import cv2


def autocorrect(inputword):
    distanceDict = {}
    inputword = inputword.lower()
    wordListweb = get_english_words_set(["web2"], lower=True)
    wordListgcide = get_english_words_set(
        ["gcide"], lower=True
    )  # Words found in GNU Collaborative International Dictionary of English 0.53.
    # with open("Data/wordList.txt") as file:
    #     wordList = [line.strip() for line in file.readlines()]
    if inputword not in wordListweb and inputword not in wordListgcide:
        for word in wordListgcide:
            distanceDict[word] = nltk.edit_distance(word, inputword)
        minval = min(distanceDict.values())
        res = list(filter(lambda x: distanceDict[x] == minval, distanceDict))
        return res
    else:
        return inputword


def findMinMax(landmarks, width, height):
    x_max, y_max, z_max = 0, 0, 0
    x_min, y_min, z_min = width, height, None
    for landmark in landmarks:
        x, y, z = landmark.x, landmark.y, landmark.z
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y
        if z_min == None or z < z_min:
            z_min = z
        if z > z_max:
            z_max = z
    return x_min, y_min, z_min, x_max, y_max, z_max


def displayMultiFrames(frame, text, x, y, display_flag):
    if display_flag:
        frame = cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
        )
        display_flag -= 1
    return display_flag
