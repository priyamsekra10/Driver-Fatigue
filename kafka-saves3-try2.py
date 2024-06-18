from kafka import KafkaConsumer
import cv2
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine
from signal import signal, SIGPIPE, SIG_DFL
import boto3
import io
from dotenv import load_dotenv
import os
import dlib
import datetime



# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
ACCESS_KEY_ID = os.getenv('ACCESS_KEY_ID')
SECRET_ACCESS_KEY = os.getenv('SECRET_ACCESS_KEY')
# debug = os.getenv('DEBUG')


s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY
)
# Initialize the S3 client
# s3_client = boto3.client('s3')
bucket_name = 'resq'  # replace with your actual bucket name
folder_name = 'video1'


def download_model_from_s3(model_key, local_path):
    s3_client.download_file(bucket_name, model_key, local_path)

# Download the ONNX models from S3 and store the paths in variables
a = "assets/face_detector.onnx"
b = "assets/face_landmarks.onnx"
d = "shape_predictor_68_face_landmarks.dat"

download_model_from_s3("driver-fatigue-models/face_detector.onnx", a)
download_model_from_s3("driver-fatigue-models/face_landmarks.onnx", b)
download_model_from_s3("driver-fatigue-models/shape_predictor_68_face_landmarks.dat", d)


# Initialize dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(d)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

driver_id = "priyam_sekra"

def update_driver_behavior(driver_id, behavior_data):
    # Initialize a session using Amazon DynamoDB with credentials
    dynamodb = boto3.resource(
        'dynamodb',
        aws_access_key_id=os.getenv('ACCESS_KEY_ID'),
        aws_secret_access_key= os.getenv('SECRET_ACCESS_KEY'),
        region_name='ap-south-1'
    )

    # Select your DynamoDB table
    table = dynamodb.Table('DriverBehavior')

    # Generate a timestamp
    timestamp = datetime.datetime.utcnow().isoformat()

    # Insert data into the table
    response = table.put_item(
       Item={
            'DriverId': driver_id,
            'Timestamp': timestamp,
            'BehaviorData': behavior_data
        }
    )

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear

# Function to process each frame for detecting drowsiness
def run_on_image(frame):
    EYE_ASPECT_RATIO_THRESHOLD = 0.3
    EYE_ASPECT_RATIO_CONSEC_FRAMES = 50
    COUNTER = 0

    frame_height, frame_width, _ = frame.shape

    # Setup face, mark, and pose detectors
    face_detector = FaceDetector(a)
    mark_detector = MarkDetector(b)
    pose_estimator = PoseEstimator(frame_width, frame_height)

    # Detect faces in the frame
    faces, _ = face_detector.detect(frame, 0.7)
    tm = cv2.TickMeter()

    if len(faces) > 0:
        tm.start()
        face = refine(faces, frame_width, frame_height, 0.15)[0]
        x1, y1, x2, y2 = face[:4].astype(int)
        patch = frame[y1:y2, x1:x2]

        # Detect facial landmarks
        marks = mark_detector.detect([patch])[0].reshape([68, 2])
        marks *= (x2 - x1)
        marks[:, 0] += x1
        marks[:, 1] += y1

        # Estimate pose
        pose = pose_estimator.solve(marks)
        tm.stop()
       
        # Visualize results
        pose_estimator.visualize(frame, pose, color=(0, 255, 0))
        pose_estimator.draw_axes(frame, pose)
        mark_detector.visualize(frame, marks, color=(0, 255, 0))
        face_detector.visualize(frame, faces)

        # Eye closed/open detection
        # Eye closed/open detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for face in faces:
            rect = dlib.rectangle(int(face[0]), int(face[1]), int(face[2]), int(face[3]))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEyeAspectRatio = eye_aspect_ratio(leftEye)
            rightEyeAspectRatio = eye_aspect_ratio(rightEye)
            eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
                COUNTER += 1
                if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                    update_driver_behavior(driver_id, "Drowsy")
                    cv2.putText(frame, "You are Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            else:
                COUNTER = 0
    cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))           
    return frame

# Function to save frame to S3
def save_frame_to_s3(frame, frame_number):
    _, buffer = cv2.imencode('.png', frame)
    frame_key = f"{folder_name}/{frame_number}.png"
    s3_client.put_object(Bucket=bucket_name, Key=frame_key, Body=buffer.tobytes())

# Function to delete frame from S3
def delete_frame_from_s3(frame_number):
    frame_key = f"{folder_name}/{frame_number}.png"
    s3_client.delete_object(Bucket=bucket_name, Key=frame_key)

# Function to receive and process video stream from Kafka consumer
def receive_and_process_stream():
    topic = 'video'
    consumer = KafkaConsumer(topic,
                             bootstrap_servers=['13.201.34.30:9092'], 
                             fetch_max_bytes=26214400,
                             max_partition_fetch_bytes=26214400)

    try:
        consumer.poll()
    except Exception as e:
        print(f"Error seeking to end: {e}")

    frame_count = 0

    for msg in consumer:
        # Convert bytes to ndarray
        nparr = np.frombuffer(msg.value, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Failed to decode frame")
            continue

        # Process the frame
        processed_frame = run_on_image(frame)

        # Display the processed frame
        cv2.imshow('Driver Fatigue Detection', processed_frame)

        # Save the frame to S3
        frame_count += 1
        save_frame_to_s3(processed_frame, frame_count)

        # Delete the frame from S3
        if frame_count > 10:
            delete_frame_from_s3(frame_count - 10)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    signal(SIGPIPE, SIG_DFL)

if __name__ == "__main__":
    receive_and_process_stream()
