from flask import Flask, render_template, request
import cv2
import base64
import numpy as np

app = Flask(__name__)

# Load the pre-trained Haarcascade face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    try:
        image_data = request.json['image_data']
        # Convert base64-encoded image data to NumPy array
        img_bytes = base64.b64decode(image_data.split(',')[1])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Process the frame as needed

        # Convert the processed frame back to base64 for sending to the client (optional)
        _, encoded_frame = cv2.imencode('.jpg', frame)
        encoded_frame_data = base64.b64encode(encoded_frame.tobytes()).decode('utf-8')

        return {'image_data': encoded_frame_data}
    except Exception as e:
        print("Error processing frame:", str(e))
        return '', 204  # No content response
