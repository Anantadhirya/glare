from typing import List
import io
from PIL import Image
from fastapi import APIRouter, HTTPException, status, File, UploadFile
from pydantic import BaseModel
from db.config import get_cursor, close_connection
from utils.get_embedding import get_embedding
from utils.store_face import store_face

router = APIRouter(
    prefix="/api/faces",
    tags=["faces"]
)


class FaceCreateResponse(BaseModel):
    """Schema for face creation response."""
    id: int
    person_id: int
    image_url: str
    verified: bool


class FaceUpdateRequest(BaseModel):
    """Schema for updating a face."""
    person_id: int
    verified: bool


class FaceDetailResponse(BaseModel):
    """Schema for face detail response."""
    id: int
    person_id: int
    person_name: str
    image_url: str | None
    verified: bool
    created_at: str


class UnverifiedFaceResponse(BaseModel):
    """Schema for unverified face response."""
    id: int
    person_id: int
    person_name: str
    image_url: str | None
    created_at: str

@router.get("/unverified/all", response_model=List[UnverifiedFaceResponse])
async def get_unverified_faces():
    """Get all unverified faces, sorted by latest created_at."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        cursor.execute("""
            SELECT 
                f.id,
                f.person_id,
                p.name as person_name,
                f.image_url,
                f.created_at
            FROM faces f
            JOIN persons p ON f.person_id = p.id
            WHERE f.verified = FALSE
            ORDER BY f.created_at DESC
        """)
        faces = cursor.fetchall()
        faces = [
            {
                "id": face["id"],
                "person_id": face["person_id"],
                "person_name": face["person_name"],
                "image_url": face["image_url"],
                "created_at": str(face["created_at"])
            }
            for face in faces
        ]
        return faces
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        close_connection(conn, cursor)


@router.get("/unverified/count")
async def get_unverified_faces_count():
    """Get the count of unverified faces."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM faces 
            WHERE verified = FALSE
        """)
        result = cursor.fetchone()
        return {"count": result['count']}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        close_connection(conn, cursor)


@router.delete("/unverified/all")
async def delete_all_unverified_faces():
    """Delete all unverified faces and their persons if they have no other faces."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        
        # Get all person_ids that have unverified faces
        cursor.execute("""
            SELECT DISTINCT person_id 
            FROM faces 
            WHERE verified = FALSE
        """)
        person_ids = [row['person_id'] for row in cursor.fetchall()]
        
        # Delete all unverified faces
        cursor.execute("DELETE FROM faces WHERE verified = FALSE")
        deleted_count = cursor.rowcount
        
        # Check each person and delete if they have no remaining faces
        deleted_persons = 0
        for person_id in person_ids:
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM faces 
                WHERE person_id = %s
            """, (person_id,))
            count_result = cursor.fetchone()
            
            if count_result['count'] == 0:
                cursor.execute("DELETE FROM persons WHERE id = %s", (person_id,))
                deleted_persons += 1
        
        conn.commit()
        return {
            "deleted_faces": deleted_count,
            "deleted_persons": deleted_persons,
            "message": f"Deleted {deleted_count} unverified face(s) and {deleted_persons} person(s)"
        }
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        close_connection(conn, cursor)


@router.get("/{face_id}", response_model=FaceDetailResponse)
async def get_face(face_id: int):
    """Get a single face's details."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        cursor.execute("""
            SELECT 
                f.id,
                f.person_id,
                p.name as person_name,
                f.image_url,
                f.verified,
                f.created_at
            FROM faces f
            JOIN persons p ON f.person_id = p.id
            WHERE f.id = %s
        """, (face_id,))
        face = cursor.fetchone()
        
        if not face:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Face with ID {face_id} not found"
            )
        
        return {
            "id": face["id"],
            "person_id": face["person_id"],
            "person_name": face["person_name"],
            "image_url": face["image_url"],
            "verified": face["verified"],
            "created_at": str(face["created_at"])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        close_connection(conn, cursor)

@router.post("", response_model=FaceCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_face(person_id: int, file: UploadFile = File(...)):
    """
    Create a new face from an uploaded image.
    
    Args:
        person_id: ID of the person this face belongs to
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        Created face details
    """
    conn, cursor = None, None
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check if person exists
        conn, cursor = get_cursor()
        cursor.execute("SELECT id FROM persons WHERE id = %s", (person_id,))
        if not cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person with ID {person_id} not found"
            )
        close_connection(conn, cursor)
        conn, cursor = None, None
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Get face embedding
        face_embedding, face_image = get_embedding(image, return_face=True)
        
        if face_embedding is None or face_image is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in the image"
            )
        
        # Store face to database and Cloudinary
        stored_person_id, image_url, _ = store_face(
            face_image=face_image,
            person_id=person_id,
            embedding=face_embedding,
            verified=True,
            verbose=True
        )
        
        if stored_person_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store face"
            )
        
        # Get the created face ID
        conn, cursor = get_cursor()
        cursor.execute("""
            SELECT id, person_id, image_url, verified
            FROM faces
            WHERE person_id = %s AND image_url = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (stored_person_id, image_url))
        created_face = cursor.fetchone()
        
        return created_face
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing image: {str(e)}"
        )
    finally:
        if conn and cursor:
            close_connection(conn, cursor)


@router.put("/{face_id}")
async def update_face(face_id: int, face: FaceUpdateRequest):
    """Update a face's person_id and verified status."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        
        # Check if face exists and get old person_id
        cursor.execute("SELECT person_id FROM faces WHERE id = %s", (face_id,))
        face_record = cursor.fetchone()
        if not face_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Face with ID {face_id} not found"
            )
        old_person_id = face_record['person_id']
        
        # Check if new person exists
        cursor.execute("SELECT id FROM persons WHERE id = %s", (face.person_id,))
        if not cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person with ID {face.person_id} not found"
            )
        
        # Update face
        cursor.execute("""
            UPDATE faces 
            SET person_id = %s, verified = %s 
            WHERE id = %s 
            RETURNING id, person_id, image_url, verified, created_at
        """, (face.person_id, face.verified, face_id))
        updated_face = cursor.fetchone()
        
        # Check if old person now has 0 faces (only if person_id changed)
        if old_person_id != face.person_id:
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM faces 
                WHERE person_id = %s
            """, (old_person_id,))
            count_result = cursor.fetchone()
            
            if count_result['count'] == 0:
                # Delete the person with no faces
                cursor.execute("DELETE FROM persons WHERE id = %s", (old_person_id,))
        
        conn.commit()
        return updated_face
    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        close_connection(conn, cursor)


@router.delete("/{face_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_face(face_id: int):
    """Delete a face and its person if no other faces remain."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        
        # Check if face exists and get person_id
        cursor.execute("SELECT id, person_id FROM faces WHERE id = %s", (face_id,))
        face_record = cursor.fetchone()
        if not face_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Face with ID {face_id} not found"
            )
        
        person_id = face_record['person_id']
        
        # Delete face
        cursor.execute("DELETE FROM faces WHERE id = %s", (face_id,))
        
        # Check if person now has 0 faces
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM faces 
            WHERE person_id = %s
        """, (person_id,))
        count_result = cursor.fetchone()
        
        if count_result['count'] == 0:
            # Delete the person with no faces
            cursor.execute("DELETE FROM persons WHERE id = %s", (person_id,))
        
        conn.commit()
        return None
    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        close_connection(conn, cursor)
