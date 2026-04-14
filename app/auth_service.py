import bcrypt
import database
import logging

def hash_password(password):
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password, hashed):
    """Verifies a password against the stored hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def register_user(username, password, role="user"):
    """Registers a new user."""
    hashed = hash_password(password)
    if database.create_user(username, hashed, role):
        logging.info(f"User {username} created successfully.")
        return True
    else:
        logging.warning(f"User {username} creation failed (already exists?).")
        return False

def login_user(username, password):
    """
    Authenticates a user.
    Returns the user dict if successful, None otherwise.
    """
    user = database.get_user(username)
    if user and verify_password(password, user['password_hash']):
        return user
    return None

def create_default_users():
    """Creates default admin and user if they don't exist."""
    # Check if admin exists
    if not database.get_user("admin"):
        logging.info("Creating default admin user...")
        register_user("admin", "admin123", "admin")
    
    if not database.get_user("testuser"):
        logging.info("Creating default test user...")
        register_user("testuser", "user123", "user")