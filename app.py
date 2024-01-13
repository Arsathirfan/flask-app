from flask import Flask, render_template, request, jsonify
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
@app.route('/video_feed', methods=['POST'])

@app.route('/video_feed', methods=['POST'])
def video_feed():
    try:
        image_data = request.json['image_data']
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Convert NumPy array elements to Python built-in types
            faces_list = [face.tolist() for face in faces]

            # If faces are detected, return the face detection coordinates
            response_data = {'faces': faces_list}
        else:
            # If no faces are detected, return an empty list
            response_data = {'faces': []}

        _, encoded_frame = cv2.imencode('.jpg', frame)
        encoded_frame_data = base64.b64encode(encoded_frame.tobytes()).decode('utf-8')

        return jsonify(response_data)

    except Exception as e:
        print("Error processing frame:", str(e))
        return '', 204  # No content response



