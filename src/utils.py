import os
import logging
import re
import json
import yaml
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DIR = DATA_DIR / "vectordb"
IMAGE_DIR = PROCESSED_DIR / "images"
DB_PATH = DATA_DIR / "tables.db"

for d in (RAW_DIR, PROCESSED_DIR, VECTOR_DIR, IMAGE_DIR):
    d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once for the application."""
    configured_level = (level or os.getenv("LOG_LEVEL") or "INFO").upper()
    numeric_level = getattr(logging, configured_level, logging.INFO)

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    else:
        root_logger.setLevel(numeric_level)


# ─────────────────────────────────────────────────────────────────────────────
# Config loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_llm_config() -> dict:
    return load_yaml(CONFIG_DIR / "llm.yaml")


def load_retriever_config() -> dict:
    return load_yaml(CONFIG_DIR / "retriever.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_json_cache(path: Path):
    """Return parsed JSON if the file exists, else None."""
    if path.exists():
        logging.getLogger(__name__).debug("Cache hit: %s", path)
        with open(path, "r") as f:
            return json.load(f)
    logging.getLogger(__name__).debug("Cache miss: %s", path)
    return None


def write_json_cache(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logging.getLogger(__name__).debug("Cache written: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_llm_output(text: str) -> str:
    """Strip chain-of-thought <think> blocks and extra whitespace."""
    text = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def split_markdown_by_section(markdown_text: str) -> list[tuple[str, str]]:
    """Split a markdown string into (section_title, content) pairs."""
    sections: list[tuple[str, str]] = []
    current_section = "General"
    buffer: list[str] = []

    for line in markdown_text.split("\n"):
        if line.startswith("#"):
            if buffer:
                sections.append((current_section, "\n".join(buffer)))
                buffer = []
            current_section = line.strip("# ").strip()
        else:
            buffer.append(line)

    if buffer:
        sections.append((current_section, "\n".join(buffer)))

    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Table markdown helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_cell(val) -> str:
    """Convert a cell value to a clean single-line string safe for GFM tables."""
    if val is None:
        return ""
    return str(val).replace("\r\n", "<br>").replace("\n", "<br>").replace("|", "\\|").strip()


def rows_to_markdown(headers: list, rows: list) -> str:
    """Build a GFM markdown table from headers + rows."""
    clean_headers = [normalize_cell(h) or f"Col{i}" for i, h in enumerate(headers)]
    separator = ["---"] * len(clean_headers)

    lines = [
        "| " + " | ".join(clean_headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        padded = (list(row) + [""] * len(clean_headers))[: len(clean_headers)]
        lines.append("| " + " | ".join(normalize_cell(c) for c in padded) + " |")

    return "\n".join(lines)