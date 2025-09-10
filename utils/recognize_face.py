from PIL import Image
import numpy as np
from utils.get_embedding import get_embedding

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def recognize_face(image: Image.Image, known_faces, threshold=0.7, top_k=5):
    """Compare test embedding with known faces and return leaderboard + result."""
    leaderboard = []
    test_emb = get_embedding(image)
    if test_emb is None:
        print("[ERROR] No face detected in test image.")
        return leaderboard, "Unknown"

    for name, emb in known_faces.items():
        sim = cosine_similarity(test_emb, emb)
        leaderboard.append((name, sim))
    # Sort descending by similarity
    leaderboard.sort(key=lambda x: x[1], reverse=True)

    # Get best match
    best_name, best_score = leaderboard[0]
    if best_score >= threshold:
        result = best_name
    else:
        result = "Unknown"

    return leaderboard[:top_k], result