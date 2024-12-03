import cv2
import os
import numpy as np
from function import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from time import sleep

# Define constants
DATA_PATH = 'MP_Data'
actions = ['A','B','C','D','E','F','G','H','I','J','K','L',
        'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  # Update this as per your actions
no_sequences = 10 # Number of sequences per action
sequence_length = 10  # Length of each sequence

# Create directories if they do not exist
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass

# Set up Mediapipe
import mediapipe as mp
mp_hands = mp.solutions.hands

# Set up the hands model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    
    # Loop through actions
    for action in actions:
        # Loop through sequences
        for sequence in range(no_sequences):
            # Loop through frames in each sequence
            for frame_num in range(sequence_length):
                # Construct the frame path
                frame_path = os.path.join('Image', action, f'{sequence}.png')
                print(f"Reading frame from path: {frame_path}")
                
                # Check if the file exists
                if not os.path.exists(frame_path):
                    print(f"File not found: {frame_path}")
                    continue

                # Read the frame
                frame = cv2.imread(frame_path)

                # Check if the frame was read correctly
                if frame is None:
                    print(f"Failed to read frame from path: {frame_path}")
                    continue

                # Make detections
                image, results = mediapipe_detection(frame, hands)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else:
                    cv2.putText(image, f'Collecting frames for {action} Video Number {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()
