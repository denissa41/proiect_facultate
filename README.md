#  Hand Gesture Recognition with Sound Feedback and Emergency Alert
  This project is the result of a PA (Procesarea Avansată a semnalelor) laboratory assignment developed at the Faculty of Automation, Computers and Electronics (FACE), University of Craiova.
For more information, visit: http://ace.ucv.ro/

  This project is a **hand gesture recognition system** using computer vision and machine learning techniques, developed in Python. It can recognize a set of predefined hand gestures via webcam and trigger audio feedback or send an emergency WhatsApp message when a specific gesture is held ("HELP").

---

## Features

-  Real-time hand detection using **MediaPipe** and **OpenCV**
-  Audio feedback based on recognized gestures (via **pygame**)
-  Machine learning support for custom gestures using **scikit-learn**
-  Gesture data persistence with model saving and reloading using `joblib`
-  Emergency alert via WhatsApp triggered by holding the "HELP" gesture
-  Add and train custom gestures live by pressing `T`

---

##  Recognized Gestures

| Gesture      | Meaning        | Action                       |
|--------------|----------------|------------------------------|
| OK           | Circle sign    | Plays `ok.mp3`               |
| Thumbs Up    | Approval       | Plays `thumbsup.mp3`         |
| Victory      | Peace sign     | Plays `victory.mp3`          |
| ILY          | Love sign      | Plays `ily.mp3`              |
| 3            | Number gesture | Plays `3.mp3`                |
| 4            | Number gesture | Plays `4.mp3`                |
| SALUT        | Open hand      | Plays `salut.mp3`            |
| HELP         | Closed fist    | Plays `help.mp3` + sends WhatsApp alert after 4 seconds |

---

## How It Works

1. **Hand landmarks** are detected using MediaPipe.
2. Gesture is either rule-based or predicted with a trained classifier.
3. Upon gesture recognition:
   - A sound is played.
   - If the gesture is "HELP", a WhatsApp message is sent after 4 seconds.

---

## Training New Gestures

1. Run the app and show your custom gesture to the camera.
2. Press **`T`** on your keyboard.
3. Enter:
   - A name for the gesture
   - A path to the corresponding `.mp3` sound
4. The model is retrained and saved automatically.

---

## Known Limitations

- Gesture accuracy may decrease with poor lighting or occluded hands.
- WhatsApp automation (via `pywhatkit`) depends on your default browser and may be unstable.
- Requires internet access to send WhatsApp messages.
- Classifier performance depends on training data quality and quantity.
