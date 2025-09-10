import os
from PIL import Image
from utils.get_embedding import get_embedding

def load_known_faces(folder_path, verbose=False):
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
                if verbose: print(f"[INFO] Registered {name}")
            else:
                if verbose: print(f"[WARN] No face found in {file}, skipping.")
    return known_faces
