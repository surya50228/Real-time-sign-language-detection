from flask import Flask, render_template, Response
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

app = Flask(__name__)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)
path = 'Model'
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# Define padding value
padding = 50  # You can adjust the padding as needed

classifier = Classifier("Model\keras_model.h5", "Model\labels.txt")
lbl=""
def generate_frames():
    global lbl  # Ensure we're modifying the global variable
    while True:
        success, img = cap.read()

        # Find the hands and their landmarks
        hands, img = detector.findHands(img)
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
            prediction, index = classifier.getPrediction(crop_img, draw=False)
            label = labels[index]
            lbl = label  # Update the global label variable

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_label')
def get_label():
    global lbl
    return lbl

if __name__ == "__main__":
    app.run(debug=True)
