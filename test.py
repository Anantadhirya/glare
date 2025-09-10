import os
import numpy as np
from PIL import Image
from utils.get_embedding import get_embedding

def load_known_faces(folder_path):
    """Load one image per person from a folder and return dict of embeddings."""
    known_faces = {}
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(file)[0]  # filename without extension
            path = os.path.join(folder_path, file)
            img = Image.open(path)
            emb = get_embedding(img, name=name)
            if emb is not None:
                known_faces[name] = emb
                print(f"[INFO] Registered {name}")
    return known_faces

def recognize_face(image: Image.Image, known_faces, threshold=0.7, top_k=5):
    """Compare test embedding with known faces and return leaderboard + result."""
    leaderboard = []
    test_emb = get_embedding(image)
    if test_emb is None:
        print("[ERROR] No face detected in test image.")
        return leaderboard, "Unknown"

    for name, emb in known_faces.items():
        sim = float(np.dot(test_emb, emb) / (np.linalg.norm(test_emb) * np.linalg.norm(emb)))
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


# ---------------------------
# Main Demo
# ---------------------------
if __name__ == "__main__":
    known_faces_dir = "known_faces"   # folder with Alice.jpg, Bob.jpg, etc.
    test_image_path = "test.jpeg"    # test image

    print("[STEP] Loading known faces...")
    known_faces = load_known_faces(known_faces_dir)

    print("[STEP] Processing test image...")
    img = Image.open(test_image_path)
    leaderboard, result = recognize_face(img, known_faces, threshold=0.7)

    print("\n=== Leaderboard (cosine similarity) ===")
    for name, score in leaderboard:
        print(f"{name:15s}: {score:.4f}")

    print("\nFinal Decision:", result)
