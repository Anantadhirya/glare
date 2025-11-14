import io
import json
import os
from PIL import Image
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

from utils.recognize_face import recognize_face
from utils.load_known_faces import load_known_faces

# Load environment variables
load_dotenv()

BROKER = os.getenv("MQTT_BROKER", "localhost")
PORT = int(os.getenv("MQTT_PORT", "1883"))
SUBSCRIBE_TOPIC = os.getenv("MQTT_SUBSCRIBE_TOPIC", "glare/image")
PUBLISH_TOPIC = os.getenv("MQTT_PUBLISH_TOPIC", "glare/result")

KNOWN_FACES_DIR = os.getenv("KNOWN_FACES_DIR", "known_faces")
THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.7"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

known_faces = load_known_faces(KNOWN_FACES_DIR, verbose=True)

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

    result, leaderboard = recognize_face(image, known_faces, threshold=THRESHOLD, top_k=TOP_K_RESULTS, verbose=True)
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