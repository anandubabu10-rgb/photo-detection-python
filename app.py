import os
import uuid
import json
import numpy as np
import cv2
import face_recognition
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Event Face Detection API")

BASE_DIR = "events_data"
MATCH_THRESHOLD = 0.5

os.makedirs(BASE_DIR, exist_ok=True)


# -----------------------------
# Utilities
# -----------------------------

def read_image(file_bytes):
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_event_paths(event_id):
    event_path = os.path.join(BASE_DIR, event_id)
    os.makedirs(event_path, exist_ok=True)

    return (
        os.path.join(event_path, "embeddings.npy"),
        os.path.join(event_path, "image_ids.json")
    )


# -----------------------------
# Upload Group Photo
# -----------------------------

@app.post("/upload")
async def upload_image(event_id: str, file: UploadFile = File(...)):

    image_bytes = await file.read()
    rgb_img = read_image(image_bytes)

    encodings = face_recognition.face_encodings(rgb_img)

    if not encodings:
        raise HTTPException(status_code=400, detail="No face detected")

    image_id = str(uuid.uuid4())

    embeddings_path, image_ids_path = get_event_paths(event_id)

    if os.path.exists(embeddings_path):
        embeddings = np.load(embeddings_path)
        with open(image_ids_path) as f:
            image_ids = json.load(f)
    else:
        embeddings = np.empty((0, 128), dtype=np.float32)
        image_ids = []

    new_embeddings = np.array(encodings, dtype=np.float32)

    embeddings = np.vstack((embeddings, new_embeddings))
    np.save(embeddings_path, embeddings)

    image_ids.extend([image_id] * len(encodings))

    with open(image_ids_path, "w") as f:
        json.dump(image_ids, f)

    return {
        "message": "Image processed",
        "event_id": event_id,
        "image_id": image_id,
        "faces_detected": len(encodings)
    }


# -----------------------------
# Search Selfie
# -----------------------------

@app.post("/search")
async def search_face(event_id: str, file: UploadFile = File(...)):

    embeddings_path, image_ids_path = get_event_paths(event_id)

    if not os.path.exists(embeddings_path):
        raise HTTPException(status_code=404, detail="Event not found")

    image_bytes = await file.read()
    rgb_img = read_image(image_bytes)

    encodings = face_recognition.face_encodings(rgb_img)

    if not encodings:
        raise HTTPException(status_code=400, detail="No face detected")

    selfie_embedding = np.array(encodings[0], dtype=np.float32)

    embeddings = np.load(embeddings_path)

    with open(image_ids_path) as f:
        image_ids = json.load(f)

    distances = np.linalg.norm(embeddings - selfie_embedding, axis=1)

    matched_indices = np.where(distances < MATCH_THRESHOLD)[0]

    matched_images = list(set([image_ids[i] for i in matched_indices]))

    return {
        "event_id": event_id,
        "matched_images": matched_images,
        "total_matches": len(matched_images)
    }


# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
def health():
    return {"status": "running"}


# -----------------------------
# Start Server
# -----------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
