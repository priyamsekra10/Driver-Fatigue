from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import cv2
import numpy as np
import os
from datetime import datetime

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
save_folder = "saved_frames"
os.makedirs(save_folder, exist_ok=True)

# Endpoint to process the received frame using an OpenCV model
@app.post("/api/process_frame/")
async def process_frame(file: UploadFile = File(...)):
    # Read the uploaded frame
    contents = await file.read()

    # Decode the frame
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform your OpenCV model processing on the frame
    # ...

    # Example: Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Save the processed frame to a file with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{save_folder}/frame_{timestamp}.jpg"
    cv2.imwrite(filename, gray_frame)

    # Return the processed frame
    _, buffer = cv2.imencode('.jpg', gray_frame)
    processed_frame_bytes = buffer.tobytes()

    return StreamingResponse(BytesIO(processed_frame_bytes), media_type="image/jpeg")
