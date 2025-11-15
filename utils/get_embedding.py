import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Face detector + aligner - keep_all to get all detection info
mtcnn = MTCNN(image_size=160, margin=0, device=device, keep_all=False)

# Face encoder (pretrained on VGGFace2)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img: Image.Image, return_face=False):
    """Detect, align, and embed a single face from an image."""
    img_rgb = img.convert("RGB")
    
    # Get the aligned face tensor for embedding
    face = mtcnn(img_rgb)
    if face is None:
        return None if not return_face else (None, None)
    
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device))
    embedding = F.normalize(embedding, p=2, dim=1).cpu().numpy()[0]

    if return_face:
        # Detect again to get bounding box for cropping original image
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is not None and len(boxes) > 0:
            x1, y1, x2, y2 = [int(b) for b in boxes[0]]
            face_pil = img_rgb.crop((x1, y1, x2, y2))
            return embedding, face_pil
        else:
            return None, None
    
    return embedding