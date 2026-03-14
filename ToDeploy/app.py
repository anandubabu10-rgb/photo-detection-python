import os
import uuid
import json
import numpy as np
import cv2
import face_recognition

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

BASE_DIR = "events_data"
MATCH_THRESHOLD = 0.5

# Ensure base directory exists
os.makedirs(BASE_DIR, exist_ok=True)


# ---------------------------------------
# Utility Functions
# ---------------------------------------

def read_image(file_bytes):
    np_img = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img


def get_event_paths(event_id):
    event_path = os.path.join(BASE_DIR, event_id)
    os.makedirs(event_path, exist_ok=True)

    embeddings_path = os.path.join(event_path, "embeddings.npy")
    image_ids_path = os.path.join(event_path, "image_ids.json")

    return embeddings_path, image_ids_path


# ---------------------------------------
# Upload Group Photo
# ---------------------------------------

@app.post("/upload")
async def upload_image(event_id: str, file: UploadFile = File(...)):
    image_bytes = await file.read()
    rgb_img = read_image(image_bytes)

    encodings = face_recognition.face_encodings(rgb_img)

    if not encodings:
        raise HTTPException(status_code=400, detail="No face detected")

    image_id = str(uuid.uuid4())

    embeddings_path, image_ids_path = get_event_paths(event_id)

    # Load existing embeddings
    if os.path.exists(embeddings_path):
        existing_embeddings = np.load(embeddings_path)
        with open(image_ids_path, "r") as f:
            image_ids = json.load(f)
    else:
        existing_embeddings = np.empty((0, 128), dtype=np.float32)
        image_ids = []

    new_embeddings = np.array(encodings, dtype=np.float32)

    # Append embeddings
    updated_embeddings = np.vstack((existing_embeddings, new_embeddings))
    np.save(embeddings_path, updated_embeddings)

    # Append image_id for each detected face
    for _ in encodings:
        image_ids.append(image_id)

    with open(image_ids_path, "w") as f:
        json.dump(image_ids, f)

    return JSONResponse({
        "message": "Image processed",
        "event_id": event_id,
        "image_id": image_id,
        "faces_detected": len(encodings)
    })


# ---------------------------------------
# Search Selfie Inside Event
# ---------------------------------------

@app.post("/search")
async def search_face(event_id: str, file: UploadFile = File(...)):
    embeddings_path, image_ids_path = get_event_paths(event_id)

    if not os.path.exists(embeddings_path):
        raise HTTPException(status_code=404, detail="Event not found")

    image_bytes = await file.read()
    rgb_img = read_image(image_bytes)

    encodings = face_recognition.face_encodings(rgb_img)

    if not encodings:
        raise HTTPException(status_code=400, detail="No face detected in selfie")

    selfie_embedding = np.array(encodings[0], dtype=np.float32)

    # Load event embeddings
    event_embeddings = np.load(embeddings_path)

    with open(image_ids_path, "r") as f:
        image_ids = json.load(f)

    # Vectorized distance computation
    distances = np.linalg.norm(event_embeddings - selfie_embedding, axis=1)

    matched_indices = np.where(distances < MATCH_THRESHOLD)[0]

    matched_images = list(set([image_ids[i] for i in matched_indices]))

    return JSONResponse({
        "event_id": event_id,
        "matched_images": matched_images,
        "total_matches": len(matched_images)
    })


# ---------------------------------------
# Health Check
# ---------------------------------------

@app.get("/")
def health():
    return {"status": "Event-based Face Service Running"}