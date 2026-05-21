"""
ingest.py
Handles all document ingestion:
  - PDF → text + structure via Docling (replaces LlamaParse + pdfplumber)
  - PDF → images via PyMuPDF (noise-filtered, attached to text chunks)
  - Tables → structured rows in SQLite + rich summary Documents for vector index
  - Builds LlamaIndex Document objects for text and table summaries
"""

import asyncio
import json
import logging
import re
import sqlite3
from pathlib import Path

import pymupdf
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import TableItem

from llama_index.core import Document

from src.utils import (
    IMAGE_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    DB_PATH,
    read_json_cache,
    write_json_cache,
    rows_to_markdown,
)
from src.generator import summarize_table

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Docling converter — module-level singleton (loads models once)
# ---------------------------------------------------------------------------

def _make_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    # Keep images embedded so PictureItem.get_image() works
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

_converter: DocumentConverter | None = None

def get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        logger.info("Initializing Docling converter (loads layout models once)")
        _converter = _make_converter()
    return _converter


# ---------------------------------------------------------------------------
# SQLite table store
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the manual_tables store if it does not exist yet."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS manual_tables (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name   TEXT    NOT NULL,
                page        INTEGER NOT NULL,
                table_index INTEGER NOT NULL,
                col_names   TEXT    NOT NULL,
                row_data    TEXT    NOT NULL,
                UNIQUE (file_name, page, table_index, row_data)
            )
        """)
        conn.commit()
    logger.info("SQLite table store ready at %s", DB_PATH)


def _insert_table_rows(
    file_name: str,
    page: int,
    table_index: int,
    headers: list[str],
    rows: list[list[str]],
) -> None:
    col_names_json = json.dumps(headers)
    records = [
        (file_name, page, table_index, col_names_json,
         json.dumps(dict(zip(headers, row))))
        for row in rows
        if any(cell.strip() for cell in row)
    ]
    with sqlite3.connect(DB_PATH) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO manual_tables
                (file_name, page, table_index, col_names, row_data)
            VALUES (?, ?, ?, ?, ?)
            """,
            records,
        )
        conn.commit()
    logger.debug(
        "Inserted %s rows for %s p%s table %s",
        len(records), file_name, page, table_index,
    )


def query_table(file_name: str, page: int, table_index: int) -> dict:
    """
    Return the full structured table as {col_names, rows}.
    Called at query time after the LLM decides a table is relevant.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            SELECT col_names, row_data
            FROM   manual_tables
            WHERE  file_name = ? AND page = ? AND table_index = ?
            ORDER  BY id
            """,
            (file_name, page, table_index),
        )
        results = cursor.fetchall()

    if not results:
        return {"col_names": [], "rows": []}

    col_names = json.loads(results[0][0])
    rows = [json.loads(r[1]) for r in results]
    return {"col_names": col_names, "rows": rows}


# ---------------------------------------------------------------------------
# Image extraction (noise-filtered, returns page → [filename] map)
# ---------------------------------------------------------------------------

def extract_images_from_pdf(pdf_path: str) -> dict[int, list[str]]:
    """
    Extract content images from *pdf_path* into IMAGE_DIR via PyMuPDF.

    Filters out:
      - Images appearing on > 30% of pages  (logos, watermarks)
      - Images smaller than 100×100 px      (icons, bullets)
      - Images with extreme aspect ratio    (banners, dividers)

    Returns { page_num: [image_filename, ...] }  (1-indexed).
    """
    logger.info("Extracting images from %s", pdf_path)
    doc = pymupdf.open(pdf_path)
    stem = Path(pdf_path).stem
    total_pages = len(doc)

    # First pass: count appearances of each xref across pages
    xref_page_count: dict[int, int] = {}
    for page_index in range(total_pages):
        for img in doc[page_index].get_images(full=True):
            xref = img[0]
            xref_page_count[xref] = xref_page_count.get(xref, 0) + 1

    page_images: dict[int, list[str]] = {}

    for page_index in range(total_pages):
        page_num = page_index + 1
        page_images[page_num] = []

        for img_index, img in enumerate(doc[page_index].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            aspect = width / height if height else 0

            if xref_page_count[xref] > total_pages * 0.3:
                logger.debug("Skip repeated image xref=%s (%s pages)", xref, xref_page_count[xref])
                continue
            if width < 100 or height < 100:
                logger.debug("Skip small image xref=%s (%sx%s)", xref, width, height)
                continue
            if aspect > 8 or aspect < 0.125:
                logger.debug("Skip banner image xref=%s aspect=%.2f", xref, aspect)
                continue

            image_name = f"{stem}_page_{page_num}_img_{img_index}.png"
            image_path = IMAGE_DIR / image_name
            with open(image_path, "wb") as f:
                f.write(base_image["image"])
            page_images[page_num].append(image_name)

    total = sum(len(v) for v in page_images.values())
    logger.info("Extracted %s content images from %s", total, pdf_path)
    return page_images


# ---------------------------------------------------------------------------
# Docling parsing: text + tables + caption→image associations
# ---------------------------------------------------------------------------

def _table_grid_to_headers_and_rows(
    table_item: TableItem,
) -> tuple[list[str], list[list[str]]]:
    """
    Convert a Docling TableItem's grid into (headers, data_rows).

    Docling marks header cells with cell.column_header=True.
    If no header row is detected, the first row is used as headers.
    """
    grid = table_item.data.grid  # list[list[TableCell]]
    if not grid:
        return [], []

    # Collect header row indices (rows where every non-empty cell is a column header)
    header_row_indices = set()
    for row_idx, row in enumerate(grid):
        if any(cell.column_header for cell in row):
            header_row_indices.add(row_idx)

    if header_row_indices:
        # Merge text from all header rows into one header list
        headers = []
        for col_idx in range(len(grid[0])):
            parts = [
                grid[r][col_idx].text.strip()
                for r in sorted(header_row_indices)
                if grid[r][col_idx].text.strip()
            ]
            headers.append(" ".join(parts) if parts else f"Col{col_idx}")
        data_rows = [
            [cell.text.strip() for cell in row]
            for row_idx, row in enumerate(grid)
            if row_idx not in header_row_indices
        ]
    else:
        # Fall back: treat first row as headers
        headers = [cell.text.strip() or f"Col{i}" for i, cell in enumerate(grid[0])]
        data_rows = [
            [cell.text.strip() for cell in row]
            for row in grid[1:]
        ]

    return headers, data_rows


def _parse_with_docling(
    file_path: str,
    filename: str,
) -> tuple[list[dict], dict[int, list[dict]], dict[int, list[str]]]:
    """
    Run Docling on *file_path* and return three structures:

      pages_data       — [{"text": markdown_str, "metadata": {...}}, ...]
                         One entry per page, markdown has tables replaced with
                         placeholders so the text chunks stay clean.

      page_table_meta  — {page_num: [{table_index, headers, sample_rows}, ...]}
                         Used to build SQLite rows and rich summaries.

      picture_page_map — {page_num: [image_filename, ...]}
                         Images from Docling's own extraction, keyed by page.
                         These are merged with PyMuPDF's map in parse_documents.
    """
    logger.info("Running Docling on %s", filename)
    result = get_converter().convert(file_path)
    doc = result.document

    # ── Build page_table_meta ────────────────────────────────────────────────
    page_table_meta: dict[int, list[dict]] = {}
    table_index_by_page: dict[int, int] = {}  # running counter per page

    for table_item in doc.tables:
        if not table_item.prov:
            continue
        page_num = table_item.prov[0].page_no

        headers, data_rows = _table_grid_to_headers_and_rows(table_item)
        if not headers:
            continue

        t_idx = table_index_by_page.get(page_num, 0)
        table_index_by_page[page_num] = t_idx + 1

        _insert_table_rows(filename, page_num, t_idx, headers, data_rows)

        page_table_meta.setdefault(page_num, []).append({
            "table_index": t_idx,
            "headers": headers,
            "sample_rows": data_rows[:3],
        })

    # ── Build picture_page_map from Docling's picture items ─────────────────
    # These supplement PyMuPDF's extraction for figures that Docling explicitly
    # identified and linked to captions.
    picture_page_map: dict[int, list[str]] = {}
    stem = Path(file_path).stem

    for pic_item in doc.pictures:
        if not pic_item.prov:
            continue
        page_num = pic_item.prov[0].page_no
        pil_img = pic_item.get_image(doc)
        if pil_img is None:
            continue

        pic_idx = len(picture_page_map.get(page_num, []))
        image_name = f"{stem}_docling_page_{page_num}_pic_{pic_idx}.png"
        image_path = IMAGE_DIR / image_name
        pil_img.save(image_path)
        picture_page_map.setdefault(page_num, []).append(image_name)

    logger.info(
        "Docling: %s tables, %s pictures across %s pages",
        sum(len(v) for v in page_table_meta.values()),
        sum(len(v) for v in picture_page_map.values()),
        len(doc.pages),
    )

    # ── Build pages_data using per-page markdown export ──────────────────────
    pages_data: list[dict] = []
    for page_no in sorted(doc.pages.keys()):
        page_md = doc.export_to_markdown(page_no=page_no)
        pages_data.append({
            "text": page_md,
            "metadata": {"file_name": filename, "page": page_no},
        })

    return pages_data, page_table_meta, picture_page_map


# ---------------------------------------------------------------------------
# Table summarization
# ---------------------------------------------------------------------------

async def _build_table_documents(
    page_table_meta: dict[int, list[dict]],
    filename: str,
) -> list[Document]:
    cache_file = PROCESSED_DIR / f"{filename}.table_summaries.json"
    cached = read_json_cache(cache_file)
    if cached is not None:
        logger.info("Using cached table summaries for %s (%s tables)", filename, len(cached))
        return _table_summary_records_to_docs(cached)

    # Flatten and summarize all tables in parallel
    all_metas = [
        (page_num, meta)
        for page_num, tables in page_table_meta.items()
        for meta in tables
    ]

    records = list(await asyncio.gather(*[
        _summarize_one_table(meta, filename, page_num)
        for page_num, meta in all_metas
    ]))

    write_json_cache(cache_file, records)
    logger.info("Saved %s table summaries for %s", len(records), filename)
    return _table_summary_records_to_docs(records)


async def _summarize_one_table(meta: dict, filename: str, page_num: int) -> dict:
    headers = meta["headers"]
    sample_rows = meta["sample_rows"]
    table_index = meta["table_index"]

    sample_md = rows_to_markdown(headers, sample_rows)

    try:
        summary = await summarize_table(sample_md)
    except Exception as e:
        logger.warning(
            "Could not summarize table p%s#%s for %s: %s",
            page_num, table_index, filename, e,
        )
        summary = f"Table with columns: {', '.join(headers)}."

    col_str = ", ".join(headers)
    sample_values = [
        str(cell).strip()
        for row in sample_rows
        for cell in row
        if cell and str(cell).strip()
    ]
    value_str = "; ".join(sample_values[:10])

    return {
        "summary": f"{summary} Columns: {col_str}. Sample values: {value_str}.",
        "file_name": filename,
        "page": page_num,
        "table_index": table_index,
    }


def _table_summary_records_to_docs(records: list[dict]) -> list[Document]:
    return [
        Document(
            text=record["summary"],
            metadata={
                "type": "table_summary",
                "file_name": record["file_name"],
                "page": record["page"],
                "table_index": record["table_index"],
            },
        )
        for record in records
    ]


# ---------------------------------------------------------------------------
# Section splitting + Document assembly
# ---------------------------------------------------------------------------

# Heading pattern used to split Docling's per-page markdown into sections
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def _split_markdown_into_sections(markdown_text: str) -> list[tuple[str, str]]:
    """
    Split markdown into (section_title, content) pairs on heading boundaries.
    Content before the first heading gets title "General".
    """
    sections: list[tuple[str, str]] = []
    current_title = "General"
    last_end = 0

    for m in _HEADING_RE.finditer(markdown_text):
        content = markdown_text[last_end:m.start()].strip()
        if content:
            sections.append((current_title, content))
        current_title = m.group(1).strip()
        last_end = m.end()

    tail = markdown_text[last_end:].strip()
    if tail:
        sections.append((current_title, tail))

    return sections


def _append_section_docs(
    documents: list[Document],
    markdown_text: str,
    base_metadata: dict,
    page_images: list[str],
) -> None:
    """
    Split one page's markdown into section-level Documents and append to *documents*.
    Each Document carries the filtered image filenames for its page.
    """
    for section_title, section_text in _split_markdown_into_sections(markdown_text):
        if not section_text.strip():
            continue
        documents.append(
            Document(
                text=section_text,
                metadata={
                    **base_metadata,
                    "section": section_title,
                    "type": "text",
                    "images": page_images,
                },
            )
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def parse_documents(data_dir=None, files=None) -> list[Document]:
    """
    Parse all PDFs in *data_dir* (defaults to RAW_DIR).

    For each PDF:
      - Runs Docling for text extraction and table structure recognition.
      - Runs PyMuPDF for noise-filtered image extraction.
      - Merges Docling picture items with PyMuPDF images per page.
      - Writes table rows to SQLite; generates rich summary Documents.
      - Splits per-page markdown into section-level text Documents with
        attached image references.

    Returns a flat list of Documents (type='text' and type='table_summary').
    """
    if data_dir is None:
        data_dir = RAW_DIR

    init_db()
    documents: list[Document] = []

    filenames = list(files) if files is not None else [
        f for f in (data_dir).iterdir() if f.suffix == ".pdf" and f.is_file()
        # iterdir gives Path objects; convert below
    ]
    # Normalise to plain filenames when data_dir iteration returns Path objects
    filenames = [
        f.name if isinstance(f, Path) else f
        for f in (data_dir.iterdir() if files is None else files)
    ]

    logger.info("Starting parse for %s candidate files in %s", len(filenames), data_dir)

    for filename in filenames:
        if not filename.endswith(".pdf"):
            continue

        logger.info("Processing %s", filename)
        file_path = str(data_dir / filename)

        # ── Image extraction via PyMuPDF ─────────────────────────────────────
        pymupdf_page_images = extract_images_from_pdf(file_path)

        # ── Docling parse (text + tables + docling pictures) ─────────────────
        docling_cache = PROCESSED_DIR / f"{filename}.docling.json"
        tables_cache = PROCESSED_DIR / f"{filename}.page_table_meta.json"

        if read_json_cache(docling_cache) is not None:
            logger.info("Using cached Docling output for %s", filename)
            pages_data = read_json_cache(docling_cache)
            page_table_meta = read_json_cache(tables_cache) or {}
            docling_page_images: dict[int, list[str]] = {}
        else:
            pages_data, page_table_meta, docling_page_images = _parse_with_docling(
                file_path, filename
            )
            write_json_cache(docling_cache, pages_data)
            write_json_cache(tables_cache, page_table_meta)

        # ── Merge image sources: prefer PyMuPDF; supplement with Docling ─────
        # PyMuPDF gives us all embedded images with noise filtering.
        # Docling picture items add figures it explicitly recognised
        # (e.g. rendered vector graphics PyMuPDF may miss).
        all_page_images: dict[int, list[str]] = {}
        all_pages = set(pymupdf_page_images) | set(docling_page_images)
        for p in all_pages:
            combined = pymupdf_page_images.get(p, []) + [
                img for img in docling_page_images.get(p, [])
                if img not in pymupdf_page_images.get(p, [])
            ]
            all_page_images[p] = combined

        # ── Table summary Documents ───────────────────────────────────────────
        table_docs = await _build_table_documents(page_table_meta, filename)
        documents.extend(table_docs)
        logger.info("Added %s table summary documents for %s", len(table_docs), filename)

        # ── Text Documents ────────────────────────────────────────────────────
        for page in pages_data:
            page_num = page["metadata"].get("page")
            page_imgs = all_page_images.get(page_num, [])
            _append_section_docs(documents, page["text"], page["metadata"], page_imgs)

    logger.info("Parsing complete: %s total documents", len(documents))
    return documents