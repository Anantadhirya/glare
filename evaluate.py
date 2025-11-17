import os
import numpy as np
from PIL import Image
from utils.get_embedding import get_embedding

# ---------------------------
# Main Demo
# ---------------------------
def evaluate_lfw(lfw_dir, pairs_file, threshold=0.7, verbose=False):
    correct = 0
    total = 0
    cnt = [[0, 0], [0, 0]]
    global count, print_every
    
    # Cache for embeddings to avoid recomputing
    embedding_cache = {}

    with open(pairs_file, 'r') as f:
        lines = f.readlines()[1:]  # skip header line

    if count is not None:
        lines = lines[:count]

    for line in lines:
        parts = line.strip().split(',')
        parts = [p for p in parts if p]  # remove empty strings
        if len(parts) == 3:  # same person
            name, idx1, idx2 = parts
            path1 = os.path.join(lfw_dir, name, f"{name}_{int(idx1):04d}.jpg")
            path2 = os.path.join(lfw_dir, name, f"{name}_{int(idx2):04d}.jpg")
            is_same = True
        else:  # different people
            name1, idx1, name2, idx2 = parts
            path1 = os.path.join(lfw_dir, name1, f"{name1}_{int(idx1):04d}.jpg")
            path2 = os.path.join(lfw_dir, name2, f"{name2}_{int(idx2):04d}.jpg")
            is_same = False

        # Check cache first, compute embedding only if not cached
        if path1 not in embedding_cache:
            emb1 = get_embedding(Image.open(path1))
            embedding_cache[path1] = emb1
        else:
            emb1 = embedding_cache[path1]
            
        if path2 not in embedding_cache:
            emb2 = get_embedding(Image.open(path2))
            embedding_cache[path2] = emb2
        else:
            emb2 = embedding_cache[path2]
        
        if emb1 is None or emb2 is None:
            if verbose:
                if emb1 is None: print(f"[WARN] No face found in {path1}, skipping.")
                if emb2 is None: print(f"[WARN] No face found in {path2}, skipping.")
            continue  # skip if detection failed

        sim = float(np.dot(emb1, emb2) / 
                    (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

        prediction = sim >= threshold
        if prediction == is_same:
            correct += 1
        total += 1
        cnt[int(is_same)][int(prediction)] += 1
        if verbose and total % print_every == 0:
            print(f"Processed {total} pairs... (cached: {len(embedding_cache)} unique images)")

    accuracy = correct / total if total > 0 else 0
    print(f"Total unique images processed: {len(embedding_cache)}")
    print(f"LFW Accuracy @ threshold {threshold}: {accuracy:.4f} ({correct}/{total})")
    print(f"Confusion Matrix:")
    print(f"                Predicted:  Same   Different")
    print(f"          Actual:  Same      {cnt[1][1]}      {cnt[1][0]}")
    print(f"                   Different {cnt[0][1]}      {cnt[0][0]}")
    return accuracy

from datetime import datetime
count = 6000
print_every = 100
if __name__ == "__main__":
    start_time = datetime.now()
    evaluate_lfw("lfw/lfw-deepfunneled", "lfw/pairs.csv", threshold=0.5, verbose=True)
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Evaluation completed in: {duration}")
    print(f"Average time per pair: {duration.total_seconds() / count:.4f} seconds")  # assuming 6000 pairs
