from PIL import Image
import numpy as np
import os
import io
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from db.config import get_cursor, close_connection
from utils.get_embedding import get_embedding

# Load environment variables
load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


def store_face(face_image: Image.Image, person_id: int = None, embedding: np.ndarray = None, verbose: bool = False):
    """
    Store a face image to Cloudinary and database.
    
    Args:
        face_image: PIL Image of the face
        person_id: ID of the person (if None, creates a new "Unknown" person)
        embedding: Face embedding vector (if None, will compute it)
        verbose: Whether to print debug information
    
    Returns:
        tuple: (person_id, image_url, embedding) or (None, None, None) on failure
    """
    conn, cursor = None, None
    try:
        # Get embedding if not provided
        if embedding is None:
            embedding = get_embedding(face_image, return_face=False)
            if embedding is None:
                if verbose:
                    print("[ERROR] Failed to get embedding for face")
                return None, None, None
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        face_image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            img_byte_arr,
            folder="glare/faces",
            resource_type="image"
        )
        image_url = upload_result.get("secure_url")
        
        # Store face to database
        conn, cursor = get_cursor(dict_cursor=False)
        
        # If person_id is None, insert a new Person with name "Unknown"
        if person_id is None:
            cursor.execute(
                "INSERT INTO persons (name) VALUES (%s) RETURNING id",
                ("Unknown",)
            )
            person_id = cursor.fetchone()[0]
        
        # Insert the face with the person_id
        cursor.execute(
            "INSERT INTO faces (person_id, image_url, embedding, verified) VALUES (%s, %s, %s, %s)",
            (person_id, image_url, embedding.tolist(), False)
        )
        
        conn.commit()
        
        if verbose:
            print(f"[INFO] Stored face to database with person_id={person_id}, image_url={image_url}")
        
        return person_id, image_url, embedding
        
    except Exception as e:
        if verbose:
            print(f"[ERROR] Failed to store face to database: {e}")
        if conn:
            conn.rollback()
        return None, None, None
    finally:
        if conn and cursor:
            close_connection(conn, cursor)
