import urllib.request
import os

weights_dir = "weights/mtcnn"
os.makedirs(weights_dir, exist_ok=True)

base_url = "https://github.com/timesler/facenet-pytorch/raw/master/data/"
weights = {
    "pnet.pt": base_url + "pnet.pt",
    "rnet.pt": base_url + "rnet.pt",
    "onet.pt": base_url + "onet.pt"
}

for name, url in weights.items():
    output_path = os.path.join(weights_dir, name)
    if os.path.exists(output_path):
        print(f"{name} already exists, skipping...")
    else:
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded {name} to {output_path}")

print("\nAll weights downloaded successfully!")
print(f"Weights are saved in: {os.path.abspath(weights_dir)}")
