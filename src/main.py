"""
main.py - FastAPI application
Owns the lifespan, pipeline orchestration, and HTTP endpoints (/upload, /query, /health).
"""

import logging

from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_index.core.node_parser import SentenceSplitter

from src.ingest import (
    parse_documents,
    build_image_metadata_store,
    build_image_documents,
    build_table_metadata_store,
    build_table_documents,
)
from src.retriever import build_index, insert_nodes, get_all_nodes, build_query_engine
from src.utils import RAW_DIR, IMAGE_DIR, VECTOR_DIR, clean_llm_output, load_retriever_config, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

cfg = load_retriever_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_query_engine = None
_index = None # /upload can insert without rebuilding from scratch


async def _build_pipeline() -> None:
    global _query_engine, _index

    logger.info("Building RAG pipeline")

    docstore_file = VECTOR_DIR / "docstore.json"
    if docstore_file.exists():
        logger.info("Existing docstore found, loading index from Qdrant + docstore")
        _index, nodes = build_index()  # load path — no re-embedding
    else:
        logger.info("No persisted docstore found, running full ingestion pipeline")
        # 1. Parse text documents
        documents, all_tables_map = await parse_documents()
        logger.info("Parsed %s text documents", len(documents))

        # 2. Describe images and add as documents
        image_metadata = await build_image_metadata_store()
        logger.info("Prepared metadata for %s images", len(image_metadata))
        documents.extend(build_image_documents(image_metadata))
        logger.info("Documents after image enrichment: %s", len(documents))

        # 3. Summarize tables and add as documents
        for filename, tables_map in all_tables_map.items():
            table_records = await build_table_metadata_store(tables_map, filename)
            documents.extend(build_table_documents(table_records))
            logger.info(
                "Added %s table documents from %s",
                len(table_records),
                filename,
            )

        # 4. Build vector index
        logger.info("Building vector index from %s documents", len(documents))
        _index, nodes = build_index(documents)

    # Assemble query engine
    logger.info("Building query engine with %s nodes", len(nodes))
    _query_engine = build_query_engine(_index, nodes)

    logger.info("RAG pipeline ready")


class QueryRequest(BaseModel):
    query: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global _query_engine, _index

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DIR / file.filename

    logger.info("Received upload for %s", file.filename)
    file_bytes = await file.read()
    with open(file_path, "wb") as f:
        f.write(file_bytes)

    logger.info("Saved %s (%s bytes), indexing incrementally", file.filename, len(file_bytes))

    # ── Parse only the newly uploaded file ───────────────────────────────────
    new_documents, new_tables_map = await parse_documents(files=[file.filename])
    logger.info("Parsed %s new text documents from %s", len(new_documents), file.filename)

    # Describe only images that belong to the new file.
    # build_image_metadata_store is cache-aware: it skips already-described
    # images and only processes new ones, so this call is safe but still
    # scoped to what is actually new via the cache key.
    image_metadata = await build_image_metadata_store()
    new_documents.extend(build_image_documents(image_metadata))
    logger.info("Documents after image enrichment: %s", len(new_documents))

    for filename, tables_map in new_tables_map.items():
        table_records = await build_table_metadata_store(tables_map, filename)
        new_documents.extend(build_table_documents(table_records))
        logger.info("Added %s table documents for %s", len(table_records), filename)

    # ── Chunk the new documents ───────────────────────────────────────────────
    splitter = SentenceSplitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )
    new_nodes = splitter.get_nodes_from_documents(new_documents)
    logger.info("Generated %s new nodes", len(new_nodes))

    # ── Load or initialise the index ─────────────────────────────────────────
    # If _index is already in memory (server restarted with existing data), reuse
    # it. Otherwise attempt to load from Qdrant + persisted docstore.  If neither
    # exists (very first upload ever), build a fresh index from the new nodes.
    if _index is not None:
        logger.info("Using in-memory index for incremental insert")
        insert_nodes(_index, new_nodes)
    else:
        docstore_file = VECTOR_DIR / "docstore.json"
        if docstore_file.exists():
            # Existing Qdrant collection + docstore — load then insert
            logger.info("Loading persisted index before incremental insert")
            _index, _ = build_index()          # loads from Qdrant + docstore
            insert_nodes(_index, new_nodes)
        else:
            # First-ever upload: build index from scratch with these nodes
            logger.warning(
                "No docstore found, building fresh index; existing Qdrant data may be replaced"
            )
            _index, _ = build_index(new_documents)

    # ── Rebuild query engine with the updated node set ────────────────────────
    # Always re-fetch from the docstore so BM25 sees every node, including ones
    # inserted in previous uploads that are not in this request's new_nodes list.
    all_nodes = get_all_nodes(_index)
    _query_engine = build_query_engine(_index, all_nodes)
    logger.info("Upload indexing complete for %s; query engine rebuilt with %s nodes", file.filename, len(all_nodes))

    return {"message": "uploaded and indexed", "file": file.filename}


@app.post("/query")
async def query_rag(req: QueryRequest):
    """Query the RAG pipeline and return answer + sources."""
    if _query_engine is None:
        logger.warning("Query attempted before pipeline was ready")
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})

    logger.info("Processing query (length=%s)", len(req.query))
    response = await _query_engine.aquery(req.query)

    images: list = []
    has_tables = False
    has_images = False
    sources: list = []

    for node in response.source_nodes:
        node_images = node.metadata.get("images", [])
        images.extend(node_images)

        text = node.get_content()
        if "|" in text and "---" in text:
            has_tables = True
        if node_images:
            has_images = True

        sources.append({
            "file_name": node.metadata.get("file_name", ""),
            "page": node.metadata.get("page", ""),
            "snippet": text[:150],
            "score": round(node.score, 3) if node.score is not None else None,
        })

    logger.info(
        "Query complete: %s sources, has_tables=%s, has_images=%s",
        len(sources),
        has_tables,
        has_images,
    )

    return JSONResponse(content={
        "answer": clean_llm_output(str(response)),
        "images": list(set(images)),
        "has_tables": has_tables,
        "has_images": has_images,
        "sources": sources,
    })


@app.get("/health")
def health():
    return {"status": "ok"}