import cv2
from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
from utils import refine

def run_on_image(frame):
    print("entered run_on_image")
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

def receive_and_process_stream():
    print("receive....")

    # Start capturing the video stream from the Flask server
    cap = cv2.VideoCapture('http://192.168.214.252:8080/video_feed')

    if not cap.isOpened():
        print("Error: Could not open stream")
        return
    print("entered run_on_image")

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame")
            break

        # Process the frame
        processed_frame = run_on_image(frame)

        # Display the processed frame
        cv2.imshow('processed_frame', processed_frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print("entered run_on_image")

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("HI")
    receive_and_process_stream()
