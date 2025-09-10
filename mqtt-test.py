import io
from PIL import Image
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "glare/image"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

# Open and compress image
with Image.open("samples/test1.jpeg") as img:
    # Optionally resize (e.g., max 800x800)
    img.thumbnail((800, 800))

    # Save into memory buffer as JPEG with lower quality
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=40)  # quality 0-100 (lower = more compression)
    data = buffer.getvalue()

print(f"Sending image ({len(data)} bytes)")
client.publish(TOPIC, data)
client.disconnect()