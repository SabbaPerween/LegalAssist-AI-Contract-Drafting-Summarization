import sqlite3
import datetime

DB_NAME = "documents.db"

def init_db():
    """Initializes the database and creates the documents table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            last_modified TIMESTAMP NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_document(title, content):
    """Saves a new document to the database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now()
    cursor.execute(
        "INSERT INTO documents (title, content, last_modified) VALUES (?, ?, ?)",
        (title, content, timestamp)
    )
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id

def update_document(doc_id, title, content):
    """Updates an existing document."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    timestamp = datetime.datetime.now()
    cursor.execute(
        "UPDATE documents SET title = ?, content = ?, last_modified = ? WHERE id = ?",
        (title, content, timestamp, doc_id)
    )
    conn.commit()
    conn.close()

def get_all_documents():
    """Retrieves all documents, ordered by last modified."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, last_modified FROM documents ORDER BY last_modified DESC")
    rows = cursor.fetchall()
    conn.close()
    return [{"id": row[0], "title": row[1], "last_modified": datetime.datetime.strptime(row[2].split('.')[0], '%Y-%m-%d %H:%M:%S')} for row in rows]

def get_document_by_id(doc_id):
    """Retrieves a single document by its ID."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "title": row[1], "content": row[2]}
    return None

def delete_document(doc_id):
    """Deletes a document by its ID."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
    conn.commit()
    conn.close()