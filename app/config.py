import os


APP_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(APP_DIR, ".env")


def load_app_env():
    """Load app-local .env values once, without overriding shell env vars."""
    if not os.path.exists(ENV_PATH):
        return

    with open(ENV_PATH, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_env(key, default=None):
    load_app_env()
    return os.getenv(key, default)


def get_int_env(key, default):
    value = get_env(key, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def get_float_env(key, default):
    value = get_env(key, str(default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
