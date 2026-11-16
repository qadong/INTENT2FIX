import hashlib


def production_patch_sha256(patch: str) -> str:
    """Canonical identity shared by delivered patches and execution evidence."""
    return hashlib.sha256((patch.strip() + "\n").encode("utf-8")).hexdigest()
