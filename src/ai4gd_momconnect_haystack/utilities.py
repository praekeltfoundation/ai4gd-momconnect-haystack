import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_json(filepath: Path) -> dict:
    """Reads JSON data from a file."""
    try:
        return json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Error loading JSON from %s: %s", filepath, e)
        raise
