import os
from config import get_connection

MIGRATIONS_DIR = os.path.join(os.path.dirname(__file__), "migrations")

def create_migrations_table(conn):
    """
    Create a table to track applied migrations.
    
    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            migration_name VARCHAR(255) NOT NULL UNIQUE,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.close()

def get_applied_migrations(conn):
    """
    Get list of already applied migrations.
    
    Args:
        conn: Database connection
    
    Returns:
        set: Set of migration names that have been applied
    """
    cursor = conn.cursor()
    cursor.execute("SELECT migration_name FROM schema_migrations")
    applied = {row[0] for row in cursor.fetchall()}
    cursor.close()
    return applied

def apply_migration(conn, migration_file):
    """
    Apply a single migration file.
    
    Args:
        conn: Database connection
        migration_file: Name of the migration file (e.g., "001_create_tables.sql")
    
    Returns:
        bool: True if successful, False otherwise
    """
    migration_path = os.path.join(MIGRATIONS_DIR, migration_file)
    
    if not os.path.exists(migration_path):
        print(f"[ERROR] Migration file not found: {migration_file}")
        return False
    
    try:
        with open(migration_path, 'r', encoding='utf-8') as f:
            sql = f.read()
        
        cursor = conn.cursor()
        cursor.execute(sql)
        
        # Record migration as applied
        cursor.execute(
            "INSERT INTO schema_migrations (migration_name) VALUES (%s)",
            (migration_file,)
        )
        
        conn.commit()
        cursor.close()
        print(f"[SUCCESS] Applied migration: {migration_file}")
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"[ERROR] Failed to apply migration {migration_file}: {e}")
        return False

def run_migrations(verbose=True):
    """
    Run all pending migrations in order.
    
    Args:
        verbose (bool): Print status messages
    
    Returns:
        tuple: (success_count, total_count)
    """
    conn = None
    try:
        conn = get_connection()
        if verbose:
            print("[INFO] Connected to database")
        
        # Create migrations tracking table
        create_migrations_table(conn)
        
        # Get applied migrations
        applied = get_applied_migrations(conn)
        if verbose:
            print(f"[INFO] Found {len(applied)} previously applied migrations")
        
        # Get all migration files
        migration_files = sorted([
            f for f in os.listdir(MIGRATIONS_DIR)
            if f.endswith('.sql')
        ])
        
        if not migration_files:
            if verbose:
                print("[INFO] No migration files found")
            return 0, 0
        
        # Apply pending migrations
        success_count = 0
        for migration_file in migration_files:
            if migration_file not in applied:
                if verbose:
                    print(f"[INFO] Applying migration: {migration_file}")
                if apply_migration(conn, migration_file):
                    success_count += 1
            else:
                if verbose:
                    print(f"[SKIP] Migration already applied: {migration_file}")
        
        if verbose:
            pending_count = len(migration_files) - len(applied)
            if pending_count > 0:
                print(f"\n[INFO] Applied {success_count}/{pending_count} pending migrations")
            else:
                print("\n[INFO] All migrations are up to date")
        
        return success_count, len(migration_files)
        
    except Exception as e:
        print(f"[ERROR] Migration failed: {e}")
        return 0, 0
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("=" * 50)
    print("Running database migrations...")
    print("=" * 50)
    run_migrations(verbose=True)
