import numpy as np
import cv2
from flask import Flask, jsonify, request, Response
import threading

# Initialize the video capture from the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create the Flask app
flaskApp = Flask(__name__)

# Define the main function that runs the webcam and Flask app
def main():
    # Start the Flask application on a separate thread
    threading.Thread(target=lambda: flaskApp.run(host='0.0.0.0', port=8080)).start()

# Generator function to stream video frames, yielding them as jpeg images
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream video from the webcam
@flaskApp.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a route for the root URL to provide a simple web page or message
@flaskApp.route('/')
def index():
    return 'Welcome to the Flask app with webcam access!'

# Define a route to handle POST requests for receiving a number
@flaskApp.route('/post_number', methods=['POST'])
def post_number():
    if request.json is None:
        return jsonify({'error': 'No JSON payload provided'}), 400
    if 'number' not in request.json:
        return jsonify({'error': 'Missing "number" in JSON payload'}), 400

    num_received = request.json['number']
    print('Number received:', num_received)
    return jsonify({'number': num_received}), 201

if __name__ == '__main__':
    main()

    