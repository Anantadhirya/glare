from PIL import Image
import numpy as np
from utils.get_embedding import get_embedding
from utils.store_face import store_face

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

    # Store to database if below threshold
    if best_score < store_threshold:
        stored_person_id, image_url, stored_emb = store_face(
            face_image=face,
            person_id=result_id,
            embedding=test_emb,
            verbose=verbose
        )
        
        if stored_person_id is not None:
            # Update result_id and result if a new person was created
            if result_id is None:
                result_id = stored_person_id
                result = "Unknown"
            
            # Add to known_faces for future recognition
            known_faces.append((result_id, result, test_emb))

    return result, leaderboard