import io
import json
import os
import threading
from contextlib import asynccontextmanager
from PIL import Image
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from utils.recognize_face import recognize_face
from utils.load_known_faces import load_known_faces

# Load environment variables
load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
SUBSCRIBE_TOPIC = os.getenv("MQTT_SUBSCRIBE_TOPIC", "glare/image")
MQTT_PUBLISH_SHORT_RESULT_TOPIC = os.getenv("MQTT_PUBLISH_SHORT_RESULT_TOPIC", "glare/result")
MQTT_PUBLISH_COMPLETE_RESULT_TOPIC = os.getenv("MQTT_PUBLISH_COMPLETE_RESULT_TOPIC", "glare/complete_result")

KNOWN_FACES_DIR = os.getenv("KNOWN_FACES_DIR", "known_faces")
THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.7"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
STORE_THRESHOLD = float(os.getenv("STORE_THRESHOLD", "0.85"))

# Global variables
known_faces = None
mqtt_client = None

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker")
        client.subscribe(SUBSCRIBE_TOPIC)
    else:
        print("MQTT connection failed:", rc)

def on_message(client, userdata, msg):
    global known_faces
    if DEBUG: print(f"Received {len(msg.payload)} bytes on {msg.topic}")
    
    try:
        image = Image.open(io.BytesIO(msg.payload))
        # image.show()
    except Exception as e:
        print("[ERROR] Failed to decode image:", e)
        return

    result, leaderboard = recognize_face(image, known_faces, threshold=THRESHOLD, top_k=TOP_K_RESULTS, store_threshold=STORE_THRESHOLD, verbose=DEBUG)
    payload = {
        "result": result,
        "leaderboard": [{"name": name, "similarity": similarity} for name, similarity in leaderboard]
    }
    client.publish(MQTT_PUBLISH_SHORT_RESULT_TOPIC, result)
    client.publish(MQTT_PUBLISH_COMPLETE_RESULT_TOPIC, json.dumps(payload))

def start_mqtt_client():
    """Start MQTT client in a separate thread."""
    global mqtt_client
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    
    mqtt_client.connect(BROKER, MQTT_PORT, 60)
    mqtt_client.loop_forever()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global known_faces
    
    # Startup
    print("Loading known faces...")
    known_faces = load_known_faces(verbose=DEBUG)
    
    # Start MQTT client in background thread
    print("Starting MQTT client...")
    mqtt_thread = threading.Thread(target=start_mqtt_client, daemon=True)
    mqtt_thread.start()
    
    yield
    
    # Shutdown
    if mqtt_client:
        mqtt_client.disconnect()
    print("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="GLARE Face Recognition API",
    description="Face recognition service with MQTT and HTTP endpoints",
    version="0.1.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint."""
    return "GLARE Face Recognition API"

@app.post("/api/recognize-face")
async def recognize_face_api(file: UploadFile = File(...)):
    """
    Recognize a face from an uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        JSON with recognition result and top matches
    """
    global known_faces
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        if DEBUG: print(f"API: Received image size: {image.size}")
        
        # Perform recognition
        result, leaderboard = recognize_face(
            image, 
            known_faces, 
            threshold=THRESHOLD, 
            top_k=TOP_K_RESULTS,
            store_threshold=STORE_THRESHOLD,
            verbose=DEBUG
        )
        
        # Format response
        response = {
            "result": result,
            "leaderboard": [
                {"name": name, "similarity": similarity} 
                for name, similarity in leaderboard
            ]
        }
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"API: Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/api/reload-known-faces")
async def reload_known_faces_api():
    global known_faces
    known_faces = load_known_faces(verbose=DEBUG)
    return {"status": "success", "message": "Known faces reloaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)