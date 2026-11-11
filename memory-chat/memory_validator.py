
import re

MAX_KEY_LENGTH = 64
MAX_VALUE_LENGTH = 500

BLOCKED_KEY_TERMS = {
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "auth_token",
    "private_key",
    "credential",
    "ssn",
    "social_security",
    "credit_card",
    "cvv"
}

ALLOWED_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


CANONICAL_KEY_ALIASES = {
    "favorite_programming_language": "favorite_language",
    "preferred_programming_language": "favorite_language",
    "programming_language_preference": "favorite_language",

    "project": "project_name",
    "current_project": "project_name",
    "test_project": "project_name",

    "cloud_preference": "preferred_cloud_platform",
    "preferred_cloud": "preferred_cloud_platform",
    "favorite_cloud_platform": "preferred_cloud_platform",
}


def validate_memory(key, value):
    if not isinstance(key, str) or not isinstance(value, str):
        return None

    key = key.strip().lower()
    value = value.strip()

    if not key or not value:
        return None

    if len(key) > MAX_KEY_LENGTH:
        return None

    if len(value) > MAX_VALUE_LENGTH:
        return None

    if not ALLOWED_KEY_PATTERN.fullmatch(key):
        return None

    if any(term in key for term in BLOCKED_KEY_TERMS):
        return None

    key = CANONICAL_KEY_ALIASES.get(key, key)

    return key, value
