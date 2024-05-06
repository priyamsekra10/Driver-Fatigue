"""Integrated code for head pose estimation and eye closed/open detection."""
from argparse import ArgumentParser
import cv2
from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import dlib
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

# Constants for drowsiness detection
EYE_ASPECT_RATIO_THRESHOLD = 0.3
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50
COUNTER = 0

# Load face cascade for drowsiness detection
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Initialize dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None, help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=0, help="The webcam index.")
args = parser.parse_args()

print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear

def run():
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)
    print(f"Video source: {video_src}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    face_detector = FaceDetector("assets/face_detector.onnx")
    mark_detector = MarkDetector("assets/face_landmarks.onnx")
    pose_estimator = PoseEstimator(frame_width, frame_height)

    tm = cv2.TickMeter()

    COUNTER = 0

    while True:
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        if args.cam == 0:
            frame = cv2.flip(frame, 2)

        faces, _ = face_detector.detect(frame, 0.7)

        if len(faces) > 0:
            tm.start()

            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            marks = mark_detector.detect([patch])[0].reshape([68, 2])

            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            pose = pose_estimator.solve(marks)

            tm.stop()

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
                        cv2.putText(frame, "You are Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                else:
                    COUNTER = 0

        cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break

if __name__ == '__main__':
    run()
