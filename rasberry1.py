from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import dlib
import cv2

# Constants for drowsiness detection
EYE_ASPECT_RATIO_THRESHOLD = 0.3
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2 * C)
    return ear

def detect_driver_fatigue(input_image):
    # Initialize dlib face detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
    
    # Convert input image to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray, 0)

    # Iterate over detected faces
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)
        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(input_image, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(input_image, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check for drowsiness
        if eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
            cv2.putText(input_image, "You are Drowsy", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

    return input_image


from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import cv2
import numpy as np
import os
from datetime import datetime

# Import the detect_driver_fatigue function
# from your_module import detect_driver_fatigue

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
save_folder = "saved_frames1"
os.makedirs(save_folder, exist_ok=True)

# Counter for serial naming
frame_counter = 1

# Endpoint to process the received frame using an OpenCV model
@app.post("/api/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    global frame_counter

    # Read the uploaded frame
    contents = await file.read()

    # Decode the frame
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform your OpenCV model processing on the frame
    processed_frame = detect_driver_fatigue(frame)

    # Save the processed frame to a file with a serial name
    filename = f"{save_folder}/{frame_counter}.jpg"
    cv2.imwrite(filename, processed_frame)

    # Increment the frame counter for the next frame
    frame_counter += 1

    # Return the processed frame
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_bytes = buffer.tobytes()

    return StreamingResponse(BytesIO(processed_frame_bytes), media_type="image/jpeg")




# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import StreamingResponse
# import cv2
# import numpy as np
# import smtplib
# from email.message import EmailMessage
# import io
# import asyncio

# app = FastAPI()

# # Function to send an email with an attachment
# def send_email_with_attachment(subject, body, to, attachment):
#     msg = EmailMessage()
#     msg['Subject'] = subject
#     msg['From'] = "priyam22rr@gmail.com"
#     msg['To'] = to
#     msg.set_content(body)

#     # Attach the processed frame
#     if attachment is not None:
#         msg.add_attachment(attachment, maintype='image', subtype='jpeg', filename='frame.jpg')
    
#     # Email sending logic (replace with your SMTP server details)
#     with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
#         smtp.login("priyam22rr@gmail.com", "mlonvvatlfnilogu")  # Use environment variables or secure methods to store credentials
#         smtp.send_message(msg)

# # Function to process the frame and return the processed frame bytes
# def process_frame_and_return_bytes(frame):
#     # Placeholder for your detect_driver_fatigue function call and processing
#     processed_frame = detect_driver_fatigue(frame)
#     # For demonstration, let's just use the original frame
#     # processed_frame = frame
    
#     # Convert the frame to bytes
#     _, buffer = cv2.imencode('.jpg', processed_frame)
#     processed_frame_bytes = buffer.tobytes()
    
#     return processed_frame_bytes

# # Endpoint to process the received frame and send it via email
# @app.post("api/process_frame_2/")
# async def process_and_email_frame(file: UploadFile = File(...)):
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
#     # Process the frame and get bytes
#     processed_frame_bytes = process_frame_and_return_bytes(frame)
    
#     # Send the processed frame as an email attachment
#     send_email_with_attachment("Drowsiness Detection Alert", "Please find the attached frame indicating drowsiness.", "vipul0592bhatia@gmail.com", processed_frame_bytes)
    
#     return {"message": "Email sent successfully with the processed frame."}

# # Function to run the periodic task
# async def periodic_email_task():
#     while True:
#         await asyncio.sleep(60)  # Wait for 60 seconds
#         # Here you would trigger the frame processing and email sending logic
#         # Since this is a standalone example, you'll need to integrate it with your actual frame receiving and processing logic

# # Start the periodic task
# # asyncio.create_task(periodic_email_task())  # Uncomment this line if you want to start the periodic task when the server starts