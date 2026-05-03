import os
import sys
import json
import asyncio
import re
import base64
import fitz  
import pdfplumber
import httpx

from ast import literal_eval

from dotenv import load_dotenv

# LlamaIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    set_global_handler
)
from llama_index.llms.vllm import Vllm
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core.llms import ChatMessage
from llama_cloud import AsyncLlamaCloud

# logging
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index.core").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
set_global_handler("simple")

CACHE_DIR = "./parsed_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

IMAGE_DIR = "../parsed_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


load_dotenv()
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

llama_cloud_client = AsyncLlamaCloud(api_key=LLAMA_CLOUD_API_KEY)
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
DEV_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
PROD_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEV_LLM_MODEL = "qwen/qwen3-32b"
PROD_LLM_MODEL = "Qwen3.5-122B-A10B-GGUF"

# initialize the LLM and embedding model
if os.getenv("APP_ENV") == "dev":
    embed_model = HuggingFaceEmbedding(model_name=DEV_EMBEDDING_MODEL)
    llm = Groq(
        model=DEV_LLM_MODEL,
        api_key=GROQ_API_KEY
    )
elif os.getenv("APP_ENV") == "prod":
    embed_model = HuggingFaceEmbedding(model_name=PROD_EMBEDDING_MODEL)
    llm = Vllm(
        model=PROD_LLM_MODEL,
        tensor_parallel_size=4,
        max_new_tokens=512,
        vllm_kwargs={"swap_space": 1, "gpu_memory_utilization": 0.5},
    )

Settings.llm = llm
Settings.embed_model = embed_model


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE DESCRIPTION  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

async def describe_image_with_llm(image_path: str) -> str:
    """Call a vision-capable LLM to generate a semantic description of an image."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        },
                        {
                            "type": "text",
                            "text": (
                                "You are analyzing an image from an equipment manual. "
                                "In 1-2 sentences, describe what this image shows "
                                "(e.g. wiring diagram, component location, warning label, "
                                "installation step, exploded-view diagram, etc.) and note "
                                "any key labels, part numbers, or values visible."
                            )
                        }
                    ]
                }],
                "max_tokens": 150,
            }
        )
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()


async def build_image_metadata_store(image_dir: str) -> dict:
    """
    Describe every image in image_dir once and cache the results.

    Returns:
        {
          "page_1_img_0.png": {
              "path": "/abs/path/to/image.png",
              "description": "A wiring diagram showing ...",
              "page": 1
          },
          ...
        }
    """
    cache_file = os.path.join(CACHE_DIR, "image_metadata.json")

    if os.path.exists(cache_file):
        print("Loading image metadata from cache...")
        with open(cache_file, "r") as f:
            return json.load(f)

    image_metadata = {}
    image_files = sorted(f for f in os.listdir(image_dir) if f.endswith(".png"))

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        print(f"  Describing {img_name}...")
        try:
            description = await describe_image_with_llm(img_path)
        except Exception as e:
            print(f"  Warning: could not describe {img_name}: {e}")
            description = "Unknown image content"

        # parse page number from filename: "page_3_img_1.png" → 3
        try:
            page_num = int(img_name.split("_")[1])
        except (IndexError, ValueError):
            page_num = -1

        image_metadata[img_name] = {
            "path": img_path,
            "description": description,
            "page": page_num,
        }

    with open(cache_file, "w") as f:
        json.dump(image_metadata, f, indent=2)

    print(f"  Saved descriptions for {len(image_metadata)} images.")
    return image_metadata


def build_image_documents(image_metadata: dict) -> list:
    """
    Turn each image into a standalone Document whose text is the
    semantic description — so the retriever can rank it like any other node.
    """
    docs = []
    for img_name, meta in image_metadata.items():
        docs.append(Document(
            text=meta["description"],       # ← embedded & searched
            metadata={
                "type": "image",
                "image_path": meta["path"],
                "image_name": img_name,
                "page": meta["page"],
            }
        ))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# TABLE SUMMARIZATION  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

async def summarize_table_with_llm(markdown_table: str) -> str:
    """Use the text LLM to generate a semantic summary of a markdown table."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": DEV_LLM_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are analyzing a table from an equipment manual. "
                            "Summarize what this table contains in 1-2 sentences: "
                            "what kind of data it holds (e.g. torque specs, part numbers, "
                            "wiring pin assignments, operating limits, error codes, etc.) "
                            "and the key column names or values. Be specific and concise."
                        )
                    },
                    {
                        "role": "user",
                        "content": markdown_table
                    }
                ],
                "max_tokens": 150,
            }
        )
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()


async def build_table_metadata_store(tables_map: dict, filename: str) -> list:
    """
    Summarize every table extracted from a PDF once and cache the results.

    Args:
        tables_map: { page_num: [markdown_table, ...], ... }
        filename:   PDF filename (used to namespace the cache key)

    Returns:
        List of dicts:
        [
          {
            "summary":    "A table listing torque specifications for ...",
            "markdown":   "| col1 | col2 | ...",
            "file_name":  "manual.pdf",
            "page":       3,
            "table_index": 0,
          },
          ...
        ]
    """
    cache_file = os.path.join(CACHE_DIR, f"{filename}.tables.json")

    if os.path.exists(cache_file):
        print(f"  Loading cached table metadata for {filename}...")
        with open(cache_file, "r") as f:
            return json.load(f)

    table_records = []

    for page_num, tables in tables_map.items():
        for idx, markdown_table in enumerate(tables):
            print(f"  Summarizing table page {page_num} #{idx}...")
            try:
                summary = await summarize_table_with_llm(markdown_table)
            except Exception as e:
                print(f"  Warning: could not summarize table p{page_num}#{idx}: {e}")
                summary = "A table from the equipment manual."

            table_records.append({
                "summary":     summary,
                "markdown":    markdown_table,  # preserved for display
                "file_name":   filename,
                "page":        page_num,
                "table_index": idx,
            })

    with open(cache_file, "w") as f:
        json.dump(table_records, f, indent=2)

    print(f"  Saved summaries for {len(table_records)} tables.")
    return table_records


def build_table_documents(table_records: list) -> list:
    """
    Turn each table into a standalone Document whose searchable text is
    its LLM summary; the raw markdown is stored in metadata for display.
    """
    docs = []
    for record in table_records:
        docs.append(Document(
            text=record["summary"],         # ← embedded & searched
            metadata={
                "type":        "table",
                "table_markdown": record["markdown"],  # ← returned to user
                "file_name":   record["file_name"],
                "page":        record["page"],
                "table_index": record["table_index"],
            }
        ))
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# PDF HELPERS  
# ─────────────────────────────────────────────────────────────────────────────

def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_name = f"page_{page_index+1}_img_{img_index}.png"
            image_path = os.path.join(IMAGE_DIR, image_name)
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            print(f"Saved {image_name}")


async def download_images(result):
    """Download images using presigned URLs from images_content_metadata."""
    if not result.images_content_metadata:
        print("  No image metadata found in result.")
        return

    async with httpx.AsyncClient() as http:
        for img_meta in result.images_content_metadata:
            if not hasattr(img_meta, 'filename') or not hasattr(img_meta, 'presigned_url'):
                continue
            img_name = img_meta.filename
            url = img_meta.presigned_url
            if not img_name or not url:
                continue
            dest = os.path.join(IMAGE_DIR, img_name)
            if os.path.exists(dest):
                continue
            try:
                print(f"  Downloading {img_name}...")
                response = await http.get(url)
                response.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"  Warning: could not download {img_name}: {e}")


def normalize_cell(val) -> str:
    """Convert a cell value to a clean single-line string safe for markdown tables."""
    if val is None:
        return ""
    return str(val).replace("\r\n", "<br>").replace("\n", "<br>").replace("|", "\\|").strip()


def rows_to_markdown(headers: list, rows: list) -> str:
    """
    Build a proper GFM markdown table from headers + rows.
    Every cell is normalized so output is always parseable by `marked`
    and the frontend TableRenderer — even when cells contain newlines or pipes.
    """
    clean_headers = [normalize_cell(h) or f"Col{i}" for i, h in enumerate(headers)]
    separator = ["---"] * len(clean_headers)

    lines = [
        "| " + " | ".join(clean_headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        # Pad short rows, truncate long ones
        padded = (list(row) + [""] * len(clean_headers))[: len(clean_headers)]
        lines.append("| " + " | ".join(normalize_cell(c) for c in padded) + " |")

    return "\n".join(lines)


def extract_tables_from_pdf(pdf_path):
    tables_per_page = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            formatted_tables = []
            for table in page_tables:
                if not table or not table[0]:
                    continue
                # Use our normalizer instead of df.to_markdown() which
                # breaks when cells contain newlines or pipe characters
                formatted_tables.append(rows_to_markdown(table[0], table[1:]))
            tables_per_page[i + 1] = formatted_tables
    return tables_per_page


def split_markdown_by_section(markdown_text):
    sections = []
    current_section = "General"
    buffer = []
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
# PARSE DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────

async def parse_documents_with_llamaparse(data_dir: str):
    documents = []
    all_tables_map = {}   # { filename: { page_num: [markdown, ...] } }

    for filename in os.listdir(data_dir):
        if not filename.endswith(".pdf"):
            continue

        cache_file = os.path.join(CACHE_DIR, f"{filename}.json")

        if os.path.exists(cache_file):
            print(f"Loading cached parse for {filename}...")
            with open(cache_file, "r") as f:
                pages = json.load(f)
            for page in pages:
                documents.append(Document(text=page["text"], metadata=page["metadata"]))
            continue

        file_path = os.path.join(data_dir, filename)
        extract_images_from_pdf(file_path)
        tables_map = extract_tables_from_pdf(file_path)
        all_tables_map[filename] = tables_map

        print(f"Uploading {filename} to LlamaCloud...")
        file_obj = await llama_cloud_client.files.create(
            file=file_path,
            purpose="parse"
        )

        print("Parsing file...")
        result = await llama_cloud_client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version="latest",
            agentic_options={
                "custom_prompt": "This is an equipment manual..."
            },
            output_options={
                "markdown": {
                    "tables": {"output_tables_as_markdown": True},
                },
                "images_to_save": ["embedded", "screenshot"],
            },
            expand=["markdown", "images_content_metadata"]
        )

        print("Saving to cache...")
        pages_to_save = []
        for page in result.markdown.pages:
            pages_to_save.append({
                "text": page.markdown,
                "metadata": {
                    "file_name": filename,
                    "page": page.page_number
                }
            })

        await download_images(result)

        with open(cache_file, "w") as f:
            json.dump(pages_to_save, f)

        for page in pages_to_save:
            page_num = page["metadata"]["page"]
            sections = split_markdown_by_section(page["text"])

            for section_title, section_text in sections:
                # NOTE: no "images" key here — images are independent nodes now
                documents.append(
                    Document(
                        text=section_text,
                        metadata={
                            "file_name": filename,
                            "page": page_num,
                            "section": section_title,
                            "type": "text",
                        }
                    )
                )

        # Tables are now handled via build_table_metadata_store / build_table_documents
        # so we do NOT add raw table text nodes here anymore.

    return documents, all_tables_map


# ─────────────────────────────────────────────────────────────────────────────
# INDEXING & RETRIEVAL  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def chunk_document(documents):
    if os.path.exists("./storage"):
        print("Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        nodes = list(index.docstore.docs.values())
        print("Done")
    else:
        splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        print("Chunking documents into nodes...")
        nodes = splitter.get_nodes_from_documents(documents)
        print("Creating new index...")
        index = VectorStoreIndex.from_documents(nodes)
        index.storage_context.persist("./storage")
        print("Done")
    return index, nodes


def hybrid_search(index, nodes):
    dense_retriever = index.as_retriever(similarity_top_k=10)
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=10,
    )
    hybrid_retriever = QueryFusionRetriever(
        [dense_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,
        mode="reciprocal_rerank",
    )
    return hybrid_retriever


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    # 1. Parse text documents (tables_map returned separately)
    documents, all_tables_map = await parse_documents_with_llamaparse("../data")

    # 2. Build semantic image metadata (described once, cached forever)
    image_metadata = await build_image_metadata_store(IMAGE_DIR)
    image_docs = build_image_documents(image_metadata)
    documents.extend(image_docs)

    # 3. Build semantic table metadata (summarized once, cached forever)
    for filename, tables_map in all_tables_map.items():
        table_records = await build_table_metadata_store(tables_map, filename)
        table_docs = build_table_documents(table_records)
        documents.extend(table_docs)

    # 4. Chunk + index everything (text, image descriptions, table summaries)
    index, nodes = chunk_document(documents)

    # 5. Hybrid retrieval
    hybrid_retriever = hybrid_search(index, nodes)

    # 6. Cohere reranker
    cohere_rerank = CohereRerank(
        api_key=COHERE_API_KEY,
        top_n=5,
    )

    # 7. Query
    print("Querying the index...")
    query_engine = RetrieverQueryEngine.from_args(
        hybrid_retriever,
        llm=llm,
        node_postprocessors=[cohere_rerank],
    )

    print("Generating response...")
    llm_response = await query_engine.aquery(
        "What is the name of this device?"
    )

    # 8. Collect results
    #    - image nodes  → surface image_path from metadata
    #    - table nodes  → surface table_markdown from metadata (NOT node.text which is just the summary)
    images = []
    tables = []

    for node in llm_response.source_nodes[:5]:
        node_type = node.metadata.get("type")

        if node_type == "image":
            images.append(node.metadata.get("image_path"))
        elif node_type == "table":
            tables.append(node.metadata.get("table_markdown"))  # raw markdown for display

    images = list(set(filter(None, images)))
    tables = list(set(filter(None, tables)))

    print("Answer:", llm_response)
    print("Relevant images:", images)
    print("\nRelevant tables:\n")
    for t in tables:
        print(t)
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())