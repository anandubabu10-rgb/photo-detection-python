from fastapi import FastAPI, UploadFile, File, HTTPException
import face_recognition
import numpy as np
import cv2

app = FastAPI(title="Face Embedding Service")


@app.post("/generate-embedding")
async def generate_embedding(file: UploadFile = File(...)):

    contents = await file.read()

    # Convert bytes to numpy array
    np_arr = np.frombuffer(contents, np.uint8)

    # Decode image
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb_img)

    if len(encodings) == 0:
        raise HTTPException(status_code=404, detail="No face detected")

    embedding = encodings[0].tolist()

    return {
        "embedding": embedding,
        "dimensions": len(embedding)
    }


@app.get("/health")
def health():
    return {"status": "running"}