from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import numpy as np
import tflite_runtime.interpreter as tflite
import math
import cv2
import mediapipe as mp
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ Loading Custom FaceNet 512D Model...")
interpreter = tflite.Interpreter(model_path="custom_facenet_512.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("⏳ Loading MediaPipe Vision AI...")
mp_face_detection = mp.solutions.face_detection
# 0.5 confidence just like the web/app logic
face_detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

class ImageData(BaseModel):
    image: str

@app.get("/wakeup")
def wakeup():
    return {"status": "Server is awake and ready! AI Models loaded."}

@app.post("/face-embedding")
def get_embedding(data: ImageData):
    try:
        # 1. Decode Base64 Image to OpenCV Format
        img_data = base64.b64decode(data.image.split(',')[1] if ',' in data.image else data.image)
        img_np = np.frombuffer(img_data, np.uint8)
        img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # 2. Detect face using MediaPipe
        results = face_detector.process(img_rgb)
        
        if not results.detections:
            raise HTTPException(status_code=400, detail="Face not detected. Ensure clear lighting.")

        # Get the first face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        # Original MediaPipe Bounding Box
        mp_x, mp_y = int(bbox.xmin * img_w), int(bbox.ymin * img_h)
        mp_w, mp_h = int(bbox.width * img_w), int(bbox.height * img_h)

        # 3. Simulate ML Kit Bounding Box (Expand & Shift exactly like JS/Dart)
        mlKitBoxW = mp_w * 1.15
        mlKitBoxH = mp_h * 1.35
        mlKitBoxX = mp_x - (mp_w * 0.075)
        mlKitBoxY = mp_y - (mp_h * 0.20)

        centerX = mlKitBoxX + (mlKitBoxW / 2)
        centerY = mlKitBoxY + (mlKitBoxH / 2)

        # 4. 1.35x Perfect Square Math
        maxSize = max(mlKitBoxW, mlKitBoxH)
        squareSize = int(maxSize * 1.35)

        if squareSize > img_w: squareSize = img_w
        if squareSize > img_h: squareSize = img_h

        x = int(centerX - squareSize / 2)
        y = int(centerY - squareSize / 2)

        # Boundary Clamping
        if x < 0: x = 0
        if y < 0: y = 0
        if x + squareSize > img_w: x = img_w - squareSize
        if y + squareSize > img_h: y = img_h - squareSize

        if x < 0:
            squareSize += x
            x = 0
        if y < 0:
            squareSize += y
            y = 0
        if squareSize > img_w: squareSize = img_w
        if squareSize > img_h: squareSize = img_h

        # Initial Crop
        cropped_face = img_rgb[y:y+squareSize, x:x+squareSize]

        # 5. Face Alignment (Rotation via Eye Keypoints)
        keypoints = detection.location_data.relative_keypoints
        if len(keypoints) >= 2:
            right_eye = keypoints[0] # Image left (Person's right)
            left_eye = keypoints[1]  # Image right (Person's left)
            
            rx, ry = int(right_eye.x * img_w), int(right_eye.y * img_h)
            lx, ly = int(left_eye.x * img_w), int(left_eye.y * img_h)
            
            dx = lx - rx
            dy = ly - ry
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            
            # Rotate if head is tilted more than 3 degrees
            if abs(angle_deg) > 3:
                center = (squareSize // 2, squareSize // 2)
                M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
                cropped_face = cv2.warpAffine(cropped_face, M, (squareSize, squareSize), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # 6. Resize to 160x160 and Normalize (/ 128.0)
        resized_face = cv2.resize(cropped_face, (160, 160))
        tensor = resized_face.astype(np.float32)
        tensor = (tensor - 127.5) / 128.0
        tensor = np.expand_dims(tensor, axis=0)

        # 7. Predict using custom_facenet_512.tflite
        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()
        embeddings = interpreter.get_tensor(output_details[0]['index'])[0]

        # 8. L2 Normalize Array to exactly match Dart
        magnitude = math.sqrt(np.sum(np.square(embeddings)))
        normalized = (embeddings / magnitude).tolist()

        return {"embedding": normalized}

    except Exception as e:
        print("Backend Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Bind to Render's Port dynamically
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
