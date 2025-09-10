from PIL import Image
from utils.load_known_faces import load_known_faces
from utils.recognize_face import recognize_face

if __name__ == "__main__":
    known_faces_dir = "known_faces"   # folder with Alice.jpg, Bob.jpg, etc.
    test_image_path = "test1.jpeg"    # test image

    print("[STEP] Loading known faces...")
    known_faces = load_known_faces(known_faces_dir, verbose=True)

    print("[STEP] Processing test image...")
    img = Image.open(test_image_path)
    result, leaderboard = recognize_face(img, known_faces, threshold=0.7, verbose=True)

    print("\n=== Leaderboard (cosine similarity) ===")
    for name, score in leaderboard:
        print(f"{name:15s}: {score:.4f}")

    print("\nFinal Decision:", result)
