from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from db.config import get_cursor, close_connection

router = APIRouter(
    prefix="/api/persons",
    tags=["persons"]
)


class PersonCreateRequest(BaseModel):
    """Schema for creating a new person."""
    name: str


class PersonUpdateRequest(BaseModel):
    """Schema for updating a person's name."""
    name: str


class PersonSimpleResponse(BaseModel):
    """Schema for simple person response (for select dropdowns)."""
    id: int
    name: str


class PersonListResponse(BaseModel):
    """Schema for person in list view with one sample face."""
    id: int
    name: str
    face_count: int
    sample_image_url: Optional[str]


class FaceInPersonDetail(BaseModel):
    """Schema for face in person detail view."""
    id: int
    image_url: Optional[str]
    verified: bool
    created_at: str


class PersonDetailResponse(BaseModel):
    """Schema for detailed person view with all faces."""
    id: int
    name: str
    created_at: str
    faces: List[FaceInPersonDetail]


@router.get("/simple", response_model=List[PersonSimpleResponse])
async def get_persons_simple():
    """Get all persons with only id and name (for select dropdowns)."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        cursor.execute("""
            SELECT id, name 
            FROM persons 
            ORDER BY name
        """)
        persons = cursor.fetchall()
        return persons
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        close_connection(conn, cursor)


@router.post("", response_model=PersonSimpleResponse, status_code=status.HTTP_201_CREATED)
async def create_person(person: PersonCreateRequest):
    """Create a new person."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        cursor.execute("""
            INSERT INTO persons (name) 
            VALUES (%s) 
            RETURNING id, name
        """, (person.name,))
        new_person = cursor.fetchone()
        conn.commit()
        return new_person
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        close_connection(conn, cursor)


@router.get("", response_model=List[PersonListResponse])
async def get_all_persons():
    """Get all persons with one sample face image per person."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        cursor.execute("""
            SELECT 
                p.id,
                p.name,
                COUNT(f.id) as face_count,
                (SELECT f2.image_url 
                 FROM faces f2 
                 WHERE f2.person_id = p.id 
                 ORDER BY f2.created_at ASC 
                 LIMIT 1) as sample_image_url
            FROM persons p
            LEFT JOIN faces f ON p.id = f.person_id
            GROUP BY p.id, p.name
            ORDER BY p.created_at DESC
        """)
        persons = cursor.fetchall()
        return persons
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )
    finally:
        close_connection(conn, cursor)


@router.get("/{person_id}", response_model=PersonDetailResponse)
async def get_person(person_id: int):
    """Get a specific person with all their face images."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        
        # Get person details
        cursor.execute("""
            SELECT id, name, created_at 
            FROM persons 
            WHERE id = %s
        """, (person_id,))
        person = cursor.fetchone()
        
        if not person:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person with ID {person_id} not found"
            )
        
        # Get all faces for this person
        cursor.execute("""
            SELECT id, image_url, verified, created_at
            FROM faces
            WHERE person_id = %s
            ORDER BY created_at DESC
        """, (person_id,))
        faces = cursor.fetchall()
        faces = [
            {
                "id": face["id"],
                "image_url": face["image_url"],
                "verified": face["verified"],
                "created_at": str(face["created_at"])
            }
            for face in faces
        ]
        
        return {
            "id": person["id"],
            "name": person["name"],
            "created_at": str(person["created_at"]),
            "faces": faces
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


@router.put("/{person_id}")
async def update_person(person_id: int, person: PersonUpdateRequest):
    """Update a person's name."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        
        # Check if person exists
        cursor.execute("SELECT id FROM persons WHERE id = %s", (person_id,))
        if not cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person with ID {person_id} not found"
            )
        
        # Update person
        cursor.execute("""
            UPDATE persons 
            SET name = %s 
            WHERE id = %s 
            RETURNING id, name, created_at
        """, (person.name, person_id))
        updated_person = cursor.fetchone()
        conn.commit()
        return updated_person
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


@router.delete("/{person_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_person(person_id: int):
    """Delete a person (and all associated faces due to CASCADE)."""
    conn, cursor = None, None
    try:
        conn, cursor = get_cursor()
        
        # Check if person exists
        cursor.execute("SELECT id FROM persons WHERE id = %s", (person_id,))
        if not cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person with ID {person_id} not found"
            )
        
        # Delete person (faces will be deleted automatically due to CASCADE)
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
