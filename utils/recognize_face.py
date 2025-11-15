from PIL import Image
import numpy as np
import os
import io
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from db.config import get_cursor, close_connection
from utils.get_embedding import get_embedding

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def recognize_face(image: Image.Image, known_faces: list[tuple[int, str, np.ndarray]], threshold=0.7, top_k=5, store_threshold=-1,verbose=False):
    """Compare test embedding with known faces and return result + leaderboard."""
    leaderboard = []
    test_emb, face = get_embedding(image, return_face=True)

    if test_emb is None:
        if verbose: print("[ERROR] No face detected in image.")
        return "Unknown", leaderboard

    for id, name, emb in known_faces:
        sim = cosine_similarity(test_emb, emb)
        leaderboard.append((id, name, sim))
    # Sort descending by similarity
    leaderboard.sort(key=lambda x: x[2], reverse=True)
    
    # Remove duplicates by ID, keeping highest similarity
    seen_ids = set()
    unique_leaderboard = []
    for id, name, sim in leaderboard:
        if id not in seen_ids:
            seen_ids.add(id)
            unique_leaderboard.append((id, name, sim))
    leaderboard = unique_leaderboard

    # Get best match
    best_id, best_name, best_score = leaderboard[0] if len(leaderboard) > 0 else (None, "Unknown", -1)
    if best_score >= threshold:
        result_id = best_id
        result = best_name
    else:
        result_id = None
        result = "Unknown"

    leaderboard = [(name, sim) for _, name, sim in leaderboard[:top_k]]

    # Store to database
    if best_score < store_threshold:
        conn, cursor = None, None
        try:
            # Store face to cloudinary, get a public image URL
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            face.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            
            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(
                img_byte_arr,
                folder="glare/faces",
                resource_type="image"
            )
            image_url = upload_result.get("secure_url")
            
            # Store face to database
            conn, cursor = get_cursor(dict_cursor=False)
            
            # if result == Unknown, insert a new Person with name "Unknown"
            if result_id == None:
                cursor.execute(
                    "INSERT INTO persons (name) VALUES (%s) RETURNING id",
                    ("Unknown",)
                )
                result_id = cursor.fetchone()[0]
            
            # then insert the face with the new person_id
            cursor.execute(
                "INSERT INTO faces (person_id, image_url, embedding, verified) VALUES (%s, %s, %s, %s)",
                (result_id, image_url, test_emb.tolist(), False)
            )

            known_faces.append((result_id, result, test_emb))
            
            conn.commit()
            
            if verbose:
                print(f"[INFO] Stored face to database with person_id={result_id}, image_url={image_url}")
                
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to store face to database: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn and cursor:
                close_connection(conn, cursor)

    return result, leaderboard