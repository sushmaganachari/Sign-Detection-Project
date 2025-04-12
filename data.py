import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import mediapipe as mp
from function import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define constants
DATA_PATH = os.path.join(os.getcwd(), 'MP_Data')  # Use absolute path
IMAGE_PATH = os.path.join(os.getcwd(), 'Image')   # Use absolute path
actions = ['A','B','C','D','E','F','G','H','I','J','K','L',
        'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
no_sequences = 10
sequence_length = 10

# Create directories
for action in actions:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# Set up the hands model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
    for action in actions:
        for sequence in range(no_sequences):
            print(f"\nProcessing {action} sequence {sequence}")
            
            # Load and verify source image once per sequence
            frame_path = os.path.join(IMAGE_PATH, action, f'{sequence}.png')
            if not os.path.exists(frame_path):
                print(f"Skipping missing file: {frame_path}")
                continue
                
            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"Failed to read: {frame_path}")
                continue

            # Process the same image multiple times for the sequence
            for frame_num in range(sequence_length):
                image, results = mediapipe_detection(frame, hands)
                if results.multi_hand_landmarks:
                    draw_styled_landmarks(image, results)
                    
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    
                    # Display progress
                    cv2.putText(image, f'{action} seq {sequence} frame {frame_num}', (15, 12),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        exit()
                else:
                    print(f"No hand landmarks detected in {frame_path}")

cv2.destroyAllWindows()
