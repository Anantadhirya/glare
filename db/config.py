import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    """
    Create and return a database connection.
    
    Returns:
        psycopg2.connection: Database connection object
    """
    return psycopg2.connect(DATABASE_URL)

def get_cursor(dict_cursor=True):
    """
    Create a database connection and return a cursor.
    
    Args:
        dict_cursor (bool): If True, return RealDictCursor (rows as dicts)
    
    Returns:
        tuple: (connection, cursor)
    """
    conn = get_connection()
    if dict_cursor:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
    else:
        cursor = conn.cursor()
    return conn, cursor

def close_connection(conn, cursor):
    """
    Close cursor and connection safely.
    
    Args:
        conn: Database connection
        cursor: Database cursor
    """
    if cursor:
        cursor.close()
    if conn:
        conn.close()
