import cv2
import os
import time
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Define padding value
padding = 50  # You can adjust the padding as needed

# Specify the folder to save screenshots
save_folder = "Signs/Z"  # Change this to your desired folder path
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Set the total number of screenshots to capture
total_screenshots = 300

# Initialize screenshot counter
screenshot_count = 0

# Flag to indicate if the countdown is active
countdown_active = False

# Countdown duration in seconds
countdown_duration = 5

# Variable to store the start time of the countdown
countdown_start_time = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    # Find the hands and their landmarks
    hands, img = detector.findHands(img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('s'):
            time.sleep(5);
    if hands:
        # Assuming max 2 hands are detected for simplicity
        h, w, _ = img.shape
        x_max, y_max = (0, 0)
        x_min, y_min = (w, h)

        for hand in hands:
            bbox = hand['bbox']  # Bounding box of each hand
            x_min = min(x_min, bbox[0])
            y_min = min(y_min, bbox[1])
            x_max = max(x_max, bbox[0] + bbox[2])  # bbox[2] is the width
            y_max = max(y_max, bbox[1] + bbox[3])  # bbox[3] is the height

        # Calculate width and height of the cropped area
        width = x_max - x_min
        height = y_max - y_min

        # Determine how much to adjust to maintain a square aspect ratio
        if width > height:
            diff = width - height
            y_min -= diff // 2
            y_max += diff // 2
        else:
            diff = height - width
            x_min -= diff // 2
            x_max += diff // 2

        # Add padding and ensure the coordinates remain within the frame boundaries
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop the image to the new square dimensions including the padding
        crop_img = img[y_min:y_max, x_min:x_max]
        cv2.imshow("Cropped Hands", crop_img)

        # Check for key press
        
        

        # Capture screenshots if countdown is not active
        if not countdown_active and screenshot_count < total_screenshots:
            screenshot_count += 1
            screenshot_name = f"Z{screenshot_count}.png"
            screenshot_path = os.path.join(save_folder, screenshot_name)
            cv2.imwrite(screenshot_path, crop_img)
            print(f"Screenshot saved as {screenshot_path}")

        # Display the original image
    cv2.imshow("Hand Tracking", img)

    if screenshot_count >= total_screenshots:
        print("Captured 300 screenshots. Exiting...")
        break
        

# Cleanup
cap.release()
cv2.destroyAllWindows()
