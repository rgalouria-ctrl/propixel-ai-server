from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import math
import cv2  # 🟢 Sirf OpenCV use kar rahe hain, No MediaPipe

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load TFLite Model
interpreter = tflite.Interpreter(model_path="mobilefacenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2. Load OpenCV Face Detector (Stable on Render)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class ImageData(BaseModel):
    image: str

@app.get("/wakeup")
def wakeup():
    return {"status": "Server is awake and ready!"}

@app.post("/face-embedding")
def get_embedding(data: ImageData):
    try:
        # Decode Base64 Image
        img_data = base64.b64decode(data.image.split(',')[1] if ',' in data.image else data.image)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img_np = np.array(img)

        # Detect face using OpenCV
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="Face not detected. Ensure clear lighting.")

        # Get the first face detected
        x, y, bw, bh = faces[0]

        # Crop and Resize logic (Exact match with JS)
        cx, cy = x + (bw // 2), y + (bh // 2)
        square_size = int(max(bw, bh) * 1.5)
        sx, sy = cx - (square_size // 2), int(cy - (square_size // 2) - (square_size * 0.05))
        
        crop_img = Image.new("RGB", (square_size, square_size), (0, 0, 0))
        box = (max(0, sx), max(0, sy), min(img.width, sx + square_size), min(img.height, sy + square_size))
        region = img.crop(box)
        crop_img.paste(region, (max(0, -sx), max(0, -sy)))
        
        crop_img = crop_img.resize((112, 112))
        tensor = np.array(crop_img, dtype=np.float32)
        tensor = (tensor - 127.5) / 127.5
        tensor = np.expand_dims(tensor, axis=0)

        # Predict using mobilefacenet.tflite
        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()
        embeddings = interpreter.get_tensor(output_details[0]['index'])[0]

        # Normalize 
        magnitude = math.sqrt(np.sum(np.square(embeddings)))
        normalized = (embeddings / magnitude).tolist()

        return {"embedding": normalized}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
