import io
import json
from PIL import Image
import paho.mqtt.client as mqtt

from utils.recognize_face import recognize_face
from utils.load_known_faces import load_known_faces

BROKER = "localhost"
PORT = 1883
SUBSCRIBE_TOPIC = "glare/image"
PUBLISH_TOPIC = "glare/result"

known_faces = load_known_faces("known_faces", verbose=True)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to broker")
        client.subscribe(SUBSCRIBE_TOPIC)
    else:
        print("Connection failed:", rc)

def on_message(client, userdata, msg):
    global known_faces
    print(f"Received {len(msg.payload)} bytes on {msg.topic}")
    
    try:
        image = Image.open(io.BytesIO(msg.payload))
        print(f"Received image size: {image.size}")
        # image.show()
    except Exception as e:
        print("Failed to decode image:", e)

    result, leaderboard = recognize_face(image, known_faces, threshold=0.7, verbose=True)
    payload = {
        "result": result,
        "leaderboard": [{"name": name, "similarity": similarity} for name, similarity in leaderboard]
    }
    client.publish(PUBLISH_TOPIC, json.dumps(payload))
    print("Published result:", payload)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)
client.loop_forever()