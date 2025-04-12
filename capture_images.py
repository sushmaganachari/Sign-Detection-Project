import cv2
import os

# Define constants
IMAGE_PATH = os.path.join(os.getcwd(), 'Image')
actions = ['A','B','C','D','E','F','G','H','I','J','K','L',
        'M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
no_sequences = 10

# Initialize webcam
cap = cv2.VideoCapture(0)

for action in actions:
    for sequence in range(no_sequences):
        # Create output directory
        os.makedirs(os.path.join(IMAGE_PATH, action), exist_ok=True)
        
        print(f'Capturing {action} sequence {sequence}')
        print('Press SPACE to capture or Q to quit')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Draw capture area
            cv2.rectangle(frame, (0,40), (300,400), (255,255,255), 2)
            cv2.putText(frame, f'Capturing {action} seq {sequence}', (15,35),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            cv2.imshow('Capture', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
            elif key == ord(' '):  # Space key
                # Save image
                cv2.imwrite(os.path.join(IMAGE_PATH, action, f'{sequence}.png'), frame)
                break

cap.release()
cv2.destroyAllWindows()