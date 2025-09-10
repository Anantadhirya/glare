import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F


# ---------------------------
# Setup: device, detector, encoder
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector + aligner
mtcnn = MTCNN(image_size=160, margin=0, device=device)

# Face encoder (pretrained on VGGFace2)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# ---------------------------
# Utility Functions
# ---------------------------
def get_embedding(img: Image.Image, name: str = None):
    """Detect, align, and embed a single face from an image."""
    face = mtcnn(img.convert("RGB"))
    if face is None:
        print(f"[WARN] No face detected{f" in {name}." if name else "."}")
        return None
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device))
    return F.normalize(embedding, p=2, dim=1).cpu().numpy()[0]  # L2 normalize