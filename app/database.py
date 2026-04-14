import sqlite3
import os
from datetime import datetime

import config


DB_NAME = os.path.join(config.APP_DIR, "support_tickets.db")

TICKET_COLUMNS = {
    "confidence_score": "REAL DEFAULT 0.0",
    "resolution_status": "TEXT DEFAULT 'unresolved'",
    "retrieval_score": "REAL DEFAULT 0.0",
    "kb_context_found": "INTEGER DEFAULT 0",
    "gap_group_key": "TEXT",
    "normalized_query": "TEXT",
    "suggested_kb_filename": "TEXT",
    "feedback_value": "TEXT",
    "feedback_at": "TIMESTAMP",
}

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def _get_existing_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}

def _ensure_ticket_columns(cursor):
    existing_columns = _get_existing_columns(cursor, "tickets")
    for column_name, column_definition in TICKET_COLUMNS.items():
        if column_name not in existing_columns:
            cursor.execute(
                f"ALTER TABLE tickets ADD COLUMN {column_name} {column_definition}"
            )

def init_db():
    """Initializes the database with the tickets table if it doesn't exist."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT,
                priority TEXT,
                user_id TEXT,
                ai_resolution TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(username)
            );
        ''')
        _ensure_ticket_columns(cursor)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL
            );
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_gap_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gap_group_key TEXT NOT NULL UNIQUE,
                normalized_query TEXT NOT NULL,
                display_query TEXT NOT NULL,
                suggested_kb_filename TEXT,
                category TEXT,
                occurrence_count INTEGER DEFAULT 0,
                latest_ticket_id INTEGER,
                latest_confidence_score REAL DEFAULT 0.0,
                avg_confidence_score REAL DEFAULT 0.0,
                first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_alert_count INTEGER DEFAULT 0,
                last_alert_status TEXT,
                last_alert_message TEXT,
                last_alert_at TIMESTAMP,
                FOREIGN KEY(latest_ticket_id) REFERENCES tickets(id)
            );
        ''')
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_tickets_user_created ON tickets(user_id, created_at DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(resolution_status, created_at DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_tickets_gap_key ON tickets(gap_group_key)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_gap_events_last_seen ON knowledge_gap_events(last_seen_at DESC)"
        )
        conn.commit()
    finally:
        conn.close()

def create_user(username, password_hash, role="user"):
    """Creates a new user in the database."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                       (username, password_hash, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # User already exists
    finally:
        conn.close()

def get_user(username):
    """Retrieves a user by username."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Database {DB_NAME} initialized successfully.")
