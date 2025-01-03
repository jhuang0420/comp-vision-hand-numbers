import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow import keras

from keras.models import load_model # type: ignore

# Load the trained model
model = load_model('./my_model.keras')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

# Define gesture labels (update based on your dataset labels)
data = pd.read_csv("landmark_data.csv")
# labels = data["label"].unique()
gesture_labels = ["0","1","2","3","4","5"]

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set webcam to 60 FPS 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            
            landmarks = np.array(landmarks).flatten()  # Flatten to 1D array
            landmarks = landmarks.reshape(-1,21,3)  # Reshape to match model input shape

            # Predict the gesture
            prediction = model.predict(landmarks, verbose=0)
            gesture_idx = np.argmax(prediction)
            gesture_name = gesture_labels[gesture_idx]
            
            # Display the gesture on the frame
            cv2.putText(frame, gesture_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
