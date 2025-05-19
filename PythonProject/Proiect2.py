import cv2
import mediapipe as mp
import pywhatkit
import time
import pygame
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

pygame.mixer.init()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

last_gesture = None
last_time = 0
cooldown = 10  # secunde între mesaje HELP

X_data, y_data = [], []
sounds_custom = {}

if os.path.exists("model.pkl"):
    clf = joblib.load("model.pkl")
    X_data = list(np.load("features.npy", allow_pickle=True))
    y_data = list(np.load("labels.npy", allow_pickle=True))
    print("[INFO] Modelul învățat a fost încărcat.")
else:
    clf = None

default_sounds = {
    "OK": "sounds/ok.mp3",
    "Thumbs Up": "sounds/thumbsup.mp3",
    "Victory": "sounds/victory.mp3",
    "ILY": "sounds/ily.mp3",
    "3": "sounds/3.mp3",
    "4": "sounds/4.mp3",
    "SALUT": "sounds/Salut.mp3",
    "HELP": "sounds/help.mp3",
}

def play_sound(gesture):
    sound_file = sounds_custom.get(gesture) or default_sounds.get(gesture)
    if sound_file and os.path.exists(sound_file):
        pygame.mixer.music.stop()  # oprește orice sunet curent
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play()

def retrain():
    global clf
    if X_data and y_data:
        clf = RandomForestClassifier()
        clf.fit(X_data, y_data)
        joblib.dump(clf, "model.pkl")
        np.save("features.npy", np.array(X_data, dtype=object))
        np.save("labels.npy", np.array(y_data, dtype=object))
        print("[INFO] Model reantrenat cu gesturi noi.")

def detect_gesture(hand_landmarks):
    tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]

    fingers = [0, 0, 0, 0, 0]
    for i, tip in enumerate(tips):
        if i == 0:
            if hand_landmarks.landmark[tip].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x:
                fingers[i] = 1
        else:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                fingers[i] = 1

    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
        return "OK"
    if fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    if fingers == [0, 1, 1, 0, 0]:
        return "Victory"
    if fingers == [1, 1, 0, 0, 1]:
        return "ILY"
    if fingers == [1, 1, 1, 0, 0]:
        return "3"
    if fingers == [0, 1, 1, 1, 1]:
        return "4"
    if sum(fingers) == 5:
        return "SALUT"
    if sum(fingers) == 0:
        return "HELP"
    return None

def extract_landmarks(hand_landmarks):
    return [lm.x for lm in hand_landmarks.landmark] + [lm.y for lm in hand_landmarks.landmark]

cap = cv2.VideoCapture(0)

help_start_time = None
help_message_sent = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = detect_gesture(hand_landmarks)

            if not gesture and clf:
                landmarks = extract_landmarks(hand_landmarks)
                try:
                    gesture = clf.predict([landmarks])[0]
                except:
                    gesture = None

            if gesture:
                cv2.putText(frame, f"Gest: {gesture}", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if gesture != last_gesture:
                    last_gesture = gesture
                    play_sound(gesture)

                if gesture == "HELP":
                    if help_start_time is None:
                        help_start_time = time.time()
                        help_message_sent = False
                    elapsed = time.time() - help_start_time

                    # Bara de progres
                    bar_x, bar_y, bar_w, bar_h = 50, 150, 300, 25
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), -1)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * min(elapsed / 4, 1)), bar_y + bar_h), (0, 0, 255), -1)
                    cv2.putText(frame, f"{int(min(elapsed / 4, 1) * 100)}%", (bar_x + bar_w + 10, bar_y + bar_h - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if elapsed >= 4 and not help_message_sent:
                        try:
                            pywhatkit.sendwhatmsg_instantly(
                                "+40746856264",
                                "🔴 A fost detectat gestul HELP! Trimite ajutor urgent!",
                                wait_time=6,
                                tab_close=True
                            )
                            print("[INFO] Mesaj WhatsApp trimis după 4 secunde de HELP.")
                            help_message_sent = True
                        except Exception as e:
                            print(f"[EROARE] Trimitere eșuată: {e}")
                else:
                    help_start_time = None
                    help_message_sent = False
            else:
                help_start_time = None
                help_message_sent = False
    else:
        help_start_time = None
        help_message_sent = False

    cv2.imshow("Detectare Gesturi", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('t'):
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = extract_landmarks(hand)
            label = input("Nume pentru gest nou: ")
            sound_path = input("Cale sunet (ex: sounds/callme.mp3): ")
            X_data.append(landmarks)
            y_data.append(label)
            sounds_custom[label] = sound_path
            retrain()
            print(f"[INFO] Gestul '{label}' a fost învățat.")
        else:
            print("[INFO] Nicio mână detectată pentru învățare.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






