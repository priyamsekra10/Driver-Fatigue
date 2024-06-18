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
import time

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
ACCESS_KEY_ID = os.getenv('ACCESS_KEY_ID')
SECRET_ACCESS_KEY = os.getenv('SECRET_ACCESS_KEY')

s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY
)
bucket_name = 'resq'
folder_name = 'video_chunks'

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

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear

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
                    cv2.putText(frame, "You are Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            else:
                COUNTER = 0
    cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))           
    return frame

def save_video_to_s3(video_buffer, video_number):
    video_key = f"{folder_name}/{video_number}.mp4"
    s3_client.put_object(Bucket=bucket_name, Key=video_key, Body=video_buffer.getvalue())

def delete_video_from_s3(video_number):
    video_key = f"{folder_name}/{video_number}.mp4"
    s3_client.delete_object(Bucket=bucket_name, Key=video_key)

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
    video_count = 0
    frame_width = 640  # Update this with actual frame width
    frame_height = 480  # Update this with actual frame height
    fps = 30  # Update this with actual fps
    video_buffer = io.BytesIO()
    video_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    start_time = time.time()

    for msg in consumer:
        # Convert bytes to ndarray
        nparr = np.frombuffer(msg.value, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("Failed to decode frame")
            continue

        # Process the frame
        processed_frame = run_on_image(frame)

        # Write the processed frame to the video writer
        video_writer.write(processed_frame)

        frame_count += 1

        # Check if 10 seconds have passed
        if time.time() - start_time >= 10:
            # Release the current video writer
            video_writer.release()

            # Save the video to S3
            with open('temp.mp4', 'rb') as f:
                video_buffer = io.BytesIO(f.read())
            video_count += 1
            save_video_to_s3(video_buffer, video_count)

            # Delete the oldest video from S3 if more than 2 videos exist
            if video_count > 2:
                delete_video_from_s3(video_count - 2)

            # Reset the video writer and start time
            video_writer = cv2.VideoWriter('temp.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            start_time = time.time()

    # Clean up
    video_writer.release()
    cv2.destroyAllWindows()
    signal(SIGPIPE, SIG_DFL)

if __name__ == "__main__":
    receive_and_process_stream()
