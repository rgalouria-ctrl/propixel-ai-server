from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import math
import mediapipe as mp

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

interpreter = tflite.Interpreter(model_path="mobilefacenet.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

class ImageData(BaseModel):
    image: str

@app.get("/wakeup")
def wakeup():
    return {"status": "Server is awake and ready!"}

@app.post("/face-embedding")
def get_embedding(data: ImageData):
    try:
        img_data = base64.b64decode(data.image.split(',')[1] if ',' in data.image else data.image)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img_np = np.array(img)

        results = face_detector.process(img_np)
        if not results.detections:
            raise HTTPException(status_code=400, detail="Face not detected")

        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = img_np.shape
        x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

        cx, cy = x + (bw // 2), y + (bh // 2)
        square_size = int(max(bw, bh) * 1.5)
        sx, sy = cx - (square_size // 2), int(cy - (square_size // 2) - (square_size * 0.05))
        
        crop_img = Image.new("RGB", (square_size, square_size), (0, 0, 0))
        box = (max(0, sx), max(0, sy), min(w, sx + square_size), min(h, sy + square_size))
        region = img.crop(box)
        crop_img.paste(region, (max(0, -sx), max(0, -sy)))
        
        crop_img = crop_img.resize((112, 112))
        tensor = np.array(crop_img, dtype=np.float32)
        tensor = (tensor - 127.5) / 127.5
        tensor = np.expand_dims(tensor, axis=0)

        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()
        embeddings = interpreter.get_tensor(output_details[0]['index'])[0]

        magnitude = math.sqrt(np.sum(np.square(embeddings)))
        normalized = (embeddings / magnitude).tolist()

        return {"embedding": normalized}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
