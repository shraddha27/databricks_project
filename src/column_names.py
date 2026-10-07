import re


def normalize_column_name(name: str) -> str:
    """Convert source headers into stable snake_case identifiers."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return re.sub(r"_+", "_", normalized)