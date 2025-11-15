import numpy as np
from db.config import get_cursor, close_connection

def load_known_faces(verbose=False):
    """
    Load face embeddings from the database.
    
    Args:
        verbose (bool): If True, print information about loaded faces
    
    Returns:
        list: List of tuples (person_id, person_name, embedding)
    """
    known_faces = []
    conn, cursor = get_cursor()
    
    try:
        # Query verified faces from the database
        cursor.execute("""
            SELECT f.person_id, f.embedding, p.name
            FROM faces f
            JOIN persons p ON f.person_id = p.id
        """)
        
        rows = cursor.fetchall()
        
        for row in rows:
            person_id = row['person_id']
            embedding = np.array(row['embedding'], dtype=np.float32)
            person_name = row['name']
            
            # Add tuple to the list
            known_faces.append((person_id, person_name, embedding))
            
            if verbose:
                print(f"[INFO] Loaded face for person_id={person_id} ({person_name})")
        
        if verbose:
            print(f"[INFO] Total faces loaded: {len(known_faces)}")
    
    finally:
        close_connection(conn, cursor)
    
    return known_faces
