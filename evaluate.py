import os
import numpy as np
from PIL import Image
from utils.get_embedding import get_embedding

# ---------------------------
# Main Demo
# ---------------------------
def evaluate_lfw(lfw_dir, pairs_file, threshold=0.7):
    correct = 0
    total = 0
    cnt = [[0, 0], [0, 0]]

    with open(pairs_file, 'r') as f:
        lines = f.readlines()[1:]  # skip header line

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

        emb1 = get_embedding(Image.open(path1), name=path1)
        emb2 = get_embedding(Image.open(path2), name=path2)
        if emb1 is None or emb2 is None:
            continue  # skip if detection failed

        sim = float(np.dot(emb1, emb2) / 
                    (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

        prediction = sim >= threshold
        if prediction == is_same:
            correct += 1
        total += 1
        cnt[int(is_same)][int(prediction)] += 1
        if(total % 1000 == 0):
            print(f"Processed {total} pairs...")

    accuracy = correct / total if total > 0 else 0
    print(f"LFW Accuracy @ threshold {threshold}: {accuracy:.4f} ({correct}/{total})")
    print(f"Confusion Matrix:")
    print(f"                Predicted:  Same   Different")
    print(f"          Actual:  Same      {cnt[1][1]}      {cnt[1][0]}")
    print(f"                   Different {cnt[0][1]}      {cnt[0][0]}")
    return accuracy

from datetime import datetime
if __name__ == "__main__":
    start_time = datetime.now()
    evaluate_lfw("lfw/lfw-deepfunneled", "lfw/pairs.csv", threshold=0.5)
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Evaluation completed in: {duration}")
    print(f"Average time per pair: {duration.total_seconds() / 6000:.4f} seconds")  # assuming 6000 pairs
