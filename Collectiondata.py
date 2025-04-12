import os
import cv2

# Create base directory and subdirectories if they don't exist
directory = 'Image/'
if not os.path.exists(directory):
    os.makedirs(directory)
for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    if not os.path.exists(directory + letter):
        os.makedirs(directory + letter)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    _, frame = cap.read()
    
    # Count existing images
    count = {letter.lower(): len(os.listdir(directory + f"/{letter}"))
             for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    
    # Display frame and ROI
    row, col = frame.shape[1], frame.shape[0]
    cv2.rectangle(frame, (0,40), (300,400), (255,255,255), 2)
    cv2.imshow("data", frame)
    roi = frame[40:400, 0:300]
    cv2.imshow("ROI", roi)
    
    # Handle key events
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == ord('q'):  # Press 'q' to quit
        break
    
    # Save images based on key press
    key_pressed = chr(interrupt & 0xFF)
    if 'a' <= key_pressed <= 'z':
        folder = key_pressed.upper()
        cv2.imwrite(f"{directory}{folder}/{count[key_pressed]}.png", roi)

cap.release()
cv2.destroyAllWindows()