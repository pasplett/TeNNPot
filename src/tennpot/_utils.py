"""Small internal helpers shared by model and dataset loading."""

import hashlib
from pathlib import Path


def file_sha256(path):
    """Hash a file without loading it entirely into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
