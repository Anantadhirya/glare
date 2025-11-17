import os
import requests
from tqdm import tqdm

def download_file(url, destination):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

if __name__ == "__main__":
    # URL for VGGFace2 pretrained weights
    url = "https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt"
    destination = "weights/inception_resnet_v1_vggface2.pt"
    
    print("Downloading InceptionResnetV1 pretrained weights (VGGFace2)...")
    download_file(url, destination)
    print(f"\nWeights saved to: {destination}")
