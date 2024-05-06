import cv2
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

def run_on_image(frame):
    # Get the frame size. This will be used by the following detectors.
    frame_height, frame_width, _ = frame.shape

    # Setup a face detector to detect human faces.
    face_detector = FaceDetector("assets/face_detector.onnx")

    # Setup a mark detector to detect landmarks.
    mark_detector = MarkDetector("assets/face_landmarks.onnx")

    # Setup a pose estimator to solve pose.
    pose_estimator = PoseEstimator(frame_width, frame_height)

    # Step 1: Get faces from the input image.
    faces, _ = face_detector.detect(frame, 0.7)

    # Any valid face found?
    if len(faces) > 0:
        # Step 2: Detect landmarks. Crop and feed the face area into the
        # mark detector. Note only the first face will be used for
        # demonstration.
        face = refine(faces, frame_width, frame_height, 0.15)[0]
        x1, y1, x2, y2 = face[:4].astype(int)
        patch = frame[y1:y2, x1:x2]

        # Run the mark detection.
        marks = mark_detector.detect([patch])[0].reshape([68, 2])

        # Convert the locations from local face area to the global image.
        marks *= (x2 - x1)
        marks[:, 0] += x1
        marks[:, 1] += y1

        # Step 3: Try pose estimation with 68 points.
        pose = pose_estimator.solve(marks)

        # All done. The best way to show the result would be drawing the
        # pose on the frame in realtime.

        # Do you want to see the pose annotation?
        pose_estimator.visualize(frame, pose, color=(0, 255, 0))

        # Do you want to see the axes?
        pose_estimator.draw_axes(frame, pose)

        # Do you want to see the marks?
        mark_detector.visualize(frame, marks, color=(0, 255, 0))

        # Do you want to see the face bounding boxes?
        face_detector.visualize(frame, faces)

    return frame




# if __name__ == '__main__':
#     image_path = "example_image.jpg"  # Provide the path to your input image
#     run_on_image(image_path)


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import cv2
import numpy as np
import os
from datetime import datetime

# Import the run_on_image function from the previous code snippet
# from your_previous_module import run_on_image

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a folder to save the frames
save_folder = "saved_frames2"
os.makedirs(save_folder, exist_ok=True)

# Counter for serial naming
frame_counter = 1

# Endpoint to process the received frame using an OpenCV model
@app.post("/api/process_frame_2/")
async def process_frame(file: UploadFile = File(...)):
    global frame_counter

    # Read the uploaded frame
    contents = await file.read()

    # Decode the frame
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform OpenCV model processing on the frame
    processed_frame = run_on_image(frame)

    # Save the processed frame to a file with a serial name
    filename = f"{save_folder}/{frame_counter}.jpg"
    cv2.imwrite(filename, processed_frame)

    # Increment the frame counter for the next frame
    frame_counter += 1

    # Return the processed frame
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_bytes = buffer.tobytes()

    return StreamingResponse(BytesIO(processed_frame_bytes), media_type="image/jpeg")
 