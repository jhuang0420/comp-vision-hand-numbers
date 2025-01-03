import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import csv

MAX_SIZE = 1000

# Write data to csv 
def save_landmark_data_to_csv(data, csv_file="landmark_data.csv"):
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ['x', 'y', 'z']]
            writer.writerow(header)
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"Saved {len(data)} landmarks to {csv_file}")
    

# Draw landmark points and lines 
def draw_landmarks_and_lines(frame, hand_landmarks):
    connections = mp.solutions.hands.HAND_CONNECTIONS
    h, w, c = frame.shape

    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    for connection in connections:
        x0, y0 = int(hand_landmarks.landmark[connection[0]].x * w), int(hand_landmarks.landmark[connection[0]].y * h)
        x1, y1 = int(hand_landmarks.landmark[connection[1]].x * w), int(hand_landmarks.landmark[connection[1]].y * h)
        cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.8)

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Set webcam to 60 FPS 

    print("Press a number key (0-9) to start automated capture. Press 'q' to quit.")
    
    capture_data = False
    target_label = None
    collected_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw_landmarks_and_lines(frame, hand_landmarks)
                
        cv2.imshow('Hand Tracking', frame)

        if capture_data: 
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Prepare the row: label + flattened list of landmarks (x, y, z)
                    row = [target_label]
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    collected_data.append(row)
                    print(f"Captured frame {len(collected_data)}/{MAX_SIZE} for label {target_label}")

            # Stop capturing after MAX_SIZE entries
            if len(collected_data) >= MAX_SIZE:
                save_landmark_data_to_csv(collected_data)
                collected_data = []
                capture_data = False
                print(f"Finished capturing data for label {target_label}")
                          
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        if ord('0') <= key <= ord('9'):
            target_label = chr(key)  # Convert key press to string (e.g., '0', '1', ..., '9')
            capture_data = True
            collected_data = []
            print(f"Started capturing data for label {target_label}")
        
        if key == ord('q'):  # Quit the program
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
