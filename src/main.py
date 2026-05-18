"""
main.py - FastAPI application
Owns the lifespan, pipeline orchestration, and HTTP endpoints (/upload, /query, /health).
"""

import os
import shutil

from fastapi import FastAPI, UploadFile, File
from fastapi.concurrency import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.ingest import (
    parse_documents,
    build_image_metadata_store,
    build_image_documents,
    build_table_metadata_store,
    build_table_documents,
)
from src.retriever import build_index, build_query_engine
from src.utils import RAW_DIR, IMAGE_DIR, VECTOR_DIR, clean_llm_output

# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if RAW_DIR.exists() and any(f.endswith(".pdf") for f in os.listdir(RAW_DIR)):
        await _build_pipeline()
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global query engine (set by _build_pipeline)
_query_engine = None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline builder
# ─────────────────────────────────────────────────────────────────────────────

async def _build_pipeline() -> None:
    global _query_engine

    # 1. Parse text documents
    documents, all_tables_map = await parse_documents()

    # 2. Describe images and add as documents
    image_metadata = await build_image_metadata_store()
    documents.extend(build_image_documents(image_metadata))

    # 3. Summarize tables and add as documents
    for filename, tables_map in all_tables_map.items():
        table_records = await build_table_metadata_store(tables_map, filename)
        documents.extend(build_table_documents(table_records))

    # 4. Build vector index
    index, nodes = build_index(documents)

    # 5. Assemble query engine
    _query_engine = build_query_engine(index, nodes)

    print("RAG pipeline ready.")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Save an uploaded PDF and rebuild the RAG pipeline."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    file_path = RAW_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    print(f"Saved {file.filename}, rebuilding pipeline...")

    # Clear persisted index so it is rebuilt with the new file included
    if VECTOR_DIR.exists():
        shutil.rmtree(VECTOR_DIR)

    await _build_pipeline()

    return {"message": "uploaded and indexed", "file": file.filename}


@app.post("/query")
async def query_rag(req: QueryRequest):
    """Query the RAG pipeline and return answer + sources."""
    if _query_engine is None:
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})

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
            "score": round(node.score, 3) if node.score else None,
        })

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