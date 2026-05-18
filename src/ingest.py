"""
ingest.py
Handles all document ingestion:
  - PDF → text via LlamaParse
  - PDF → images via PyMuPDF
  - PDF → tables via pdfplumber
  - Image description cache (uses generator.describe_image)
  - Table summary cache (uses generator.summarize_table)
  - Builds LlamaIndex Document objects for text, images, and tables
"""

import os
import asyncio

import fitz          # PyMuPDF
import httpx
import pdfplumber

from dotenv import load_dotenv
from llama_cloud import AsyncLlamaCloud
from llama_index.core import Document

from src.utils import (
    IMAGE_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    read_json_cache,
    write_json_cache,
    rows_to_markdown,
    split_markdown_by_section,
)
from src.generator import describe_image, summarize_table

load_dotenv()

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
llama_cloud_client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

_CACHE_DIR = PROCESSED_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Image extraction & description
# ─────────────────────────────────────────────────────────────────────────────

def extract_images_from_pdf(pdf_path: str) -> None:
    """Save every embedded image from *pdf_path* into IMAGE_DIR."""
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_name = f"page_{page_index + 1}_img_{img_index}.png"
            image_path = IMAGE_DIR / image_name
            with open(image_path, "wb") as f:
                f.write(base_image["image"])
            print(f"  Saved {image_name}")


async def _download_images(result) -> None:
    """Download images using presigned URLs from LlamaParse result."""
    if not result.images_content_metadata:
        return

    async with httpx.AsyncClient() as http:
        for img_meta in result.images_content_metadata:
            if not hasattr(img_meta, "filename") or not hasattr(img_meta, "presigned_url"):
                continue
            img_name = img_meta.filename
            url = img_meta.presigned_url
            if not img_name or not url:
                continue
            dest = IMAGE_DIR / img_name
            if dest.exists():
                continue
            try:
                print(f"  Downloading {img_name}...")
                response = await http.get(url)
                response.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"  Warning: could not download {img_name}: {e}")


async def build_image_metadata_store() -> dict:
    """
    Describe every image in IMAGE_DIR once and cache the results.

    Returns:
        { "page_1_img_0.png": {"path": ..., "description": ..., "page": ...}, ... }
    """
    cache_file = _CACHE_DIR / "image_metadata.json"
    cached = read_json_cache(cache_file)
    if cached is not None:
        print("Loading image metadata from cache...")
        return cached

    image_metadata: dict = {}
    image_files = sorted(f for f in os.listdir(IMAGE_DIR) if f.endswith(".png"))

    for img_name in image_files:
        img_path = str(IMAGE_DIR / img_name)
        print(f"  Describing {img_name}...")
        try:
            descriptions = await asyncio.gather(describe_image(img_path))
            description = descriptions[0]
        except Exception as e:
            print(f"  Warning: could not describe {img_name}: {e}")
            description = "Unknown image content"

        try:
            page_num = int(img_name.split("_")[1])
        except (IndexError, ValueError):
            page_num = -1

        image_metadata[img_name] = {
            "path": img_path,
            "description": description,
            "page": page_num,
        }

    write_json_cache(cache_file, image_metadata)
    print(f"  Saved descriptions for {len(image_metadata)} images.")
    return image_metadata


def build_image_documents(image_metadata: dict) -> list[Document]:
    """Turn each image into a Document whose text is its semantic description."""
    return [
        Document(
            text=meta["description"],
            metadata={
                "type": "image",
                "image_path": meta["path"],
                "image_name": img_name,
                "page": meta["page"],
            },
        )
        for img_name, meta in image_metadata.items()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Table extraction & summarization
# ─────────────────────────────────────────────────────────────────────────────

def extract_tables_from_pdf(pdf_path: str) -> dict:
    """Return { page_num: [markdown_table, ...], ... } for all pages."""
    tables_per_page: dict = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            formatted_tables = []
            for table in page.extract_tables() or []:
                if not table or not table[0]:
                    continue
                formatted_tables.append(rows_to_markdown(table[0], table[1:]))
            tables_per_page[i + 1] = formatted_tables
    return tables_per_page


async def build_table_metadata_store(tables_map: dict, filename: str) -> list[dict]:
    """
    Summarize every table for *filename* once and cache the results.

    Returns a list of dicts with keys: summary, markdown, file_name, page, table_index.
    """
    cache_file = _CACHE_DIR / f"{filename}.tables.json"
    cached = read_json_cache(cache_file)
    if cached is not None:
        print(f"  Loading cached table metadata for {filename}...")
        return cached

    table_records: list[dict] = []

    for page_num, tables in tables_map.items():
        for idx, markdown_table in enumerate(tables):
            print(f"  Summarizing table page {page_num} #{idx}...")
            try:
                summaries = await asyncio.gather(summarize_table(markdown_table))
                summary = summaries[0]
            except Exception as e:
                print(f"  Warning: could not summarize table p{page_num}#{idx}: {e}")
                summary = "A table from the equipment manual."

            table_records.append({
                "summary": summary,
                "markdown": markdown_table,
                "file_name": filename,
                "page": page_num,
                "table_index": idx,
            })

    write_json_cache(cache_file, table_records)
    print(f"  Saved summaries for {len(table_records)} tables.")
    return table_records


def build_table_documents(table_records: list[dict]) -> list[Document]:
    """Turn each table record into a Document whose searchable text is its summary."""
    return [
        Document(
            text=record["summary"],
            metadata={
                "type": "table",
                "table_markdown": record["markdown"],
                "file_name": record["file_name"],
                "page": record["page"],
                "table_index": record["table_index"],
            },
        )
        for record in table_records
    ]


# ─────────────────────────────────────────────────────────────────────────────
# PDF parsing via LlamaParse
# ─────────────────────────────────────────────────────────────────────────────

async def parse_documents(data_dir=None) -> tuple[list[Document], dict]:
    """
    Parse all PDFs in *data_dir* (defaults to RAW_DIR).

    Returns:
        (text_documents, all_tables_map)
        where all_tables_map = { filename: { page_num: [markdown, ...] } }
    """
    if data_dir is None:
        data_dir = RAW_DIR

    documents: list[Document] = []
    all_tables_map: dict = {}

    for filename in os.listdir(data_dir):
        if not filename.endswith(".pdf"):
            continue

        cache_file = _CACHE_DIR / f"{filename}.json"
        tables_cache_file = _CACHE_DIR / f"{filename}.tables_map.json"

        cached_pages = read_json_cache(cache_file)
        if cached_pages is not None:
            print(f"Loading cached parse for {filename}...")
            for page in cached_pages:
                _append_section_docs(documents, page["text"], page["metadata"])

            cached_tables = read_json_cache(tables_cache_file)
            if cached_tables is not None:
                all_tables_map[filename] = cached_tables
            continue

        file_path = str(data_dir / filename)

        # Extract images and tables locally
        extract_images_from_pdf(file_path)
        tables_map = extract_tables_from_pdf(file_path)
        all_tables_map[filename] = tables_map
        write_json_cache(tables_cache_file, tables_map)

        # Parse text with LlamaParse
        print(f"Uploading {filename} to LlamaCloud...")
        file_obj = await llama_cloud_client.files.create(file=file_path, purpose="parse")

        print("Parsing file...")
        result = await llama_cloud_client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version="latest",
            agentic_options={"custom_prompt": "This is an equipment manual..."},
            output_options={
                "markdown": {"tables": {"output_tables_as_markdown": True}},
                "images_to_save": ["embedded", "screenshot"],
            },
            expand=["markdown", "images_content_metadata"],
        )

        pages_to_save = [
            {
                "text": page.markdown,
                "metadata": {"file_name": filename, "page": page.page_number},
            }
            for page in result.markdown.pages
        ]

        await _download_images(result)
        write_json_cache(cache_file, pages_to_save)

        for page in pages_to_save:
            _append_section_docs(documents, page["text"], page["metadata"])

    return documents, all_tables_map


def _append_section_docs(
    documents: list[Document],
    markdown_text: str,
    base_metadata: dict,
) -> None:
    """Split markdown into sections and append one Document per section."""
    for section_title, section_text in split_markdown_by_section(markdown_text):
        documents.append(
            Document(
                text=section_text,
                metadata={
                    **base_metadata,
                    "section": section_title,
                    "type": "text",
                },
            )
        )