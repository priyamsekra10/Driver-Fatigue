import numpy as np
import cv2
from flask import Flask, Response
import threading
import boto3

# Initialize the video capture from the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Create the Flask app
flaskApp = Flask(__name__)

# Initialize AWS Kinesis client
kinesis_client = boto3.client('driver-fatigue-1', region_name='ap-south-1')

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
            frame_bytes = buffer.tobytes()
            
            # Send the frame to Kinesis
            kinesis_client.put_record(
                StreamName='your_kinesis_stream_name',
                Data=frame_bytes,
                PartitionKey='partition_key'
            )
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route to stream video from the webcam
@flaskApp.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Define a route for the root URL to provide a simple web page or message
@flaskApp.route('/')
def index():
    return 'Welcome to the Flask app with webcam access!'

if __name__ == '__main__':
    main()
