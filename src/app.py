import os
import re

from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from pydantic import BaseModel

from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from main import parse_documents_with_llamaparse, llm, hybrid_search, chunk_document

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting up...")

    yield

    # Shutdown logic
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "../parsed_images")
DATA_DIR = os.path.join(BASE_DIR, "../data")

app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

query_engine = None


# ---------------------------
# Shared helper: build/rebuild the RAG pipeline
# ---------------------------
async def build_pipeline():
    global query_engine

    from llama_index.postprocessor.cohere_rerank import CohereRerank
    from main import (
        parse_documents_with_llamaparse,
        llm,
        hybrid_search,
        chunk_document,
        build_image_metadata_store,
        build_image_documents,
        build_table_metadata_store,
        build_table_documents,
        IMAGE_DIR,
    )

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    # Parse documents
    documents, all_tables_map = await parse_documents_with_llamaparse(DATA_DIR)

    # Build image docs
    image_metadata = await build_image_metadata_store(IMAGE_DIR)
    image_docs = build_image_documents(image_metadata)
    documents.extend(image_docs)

    # Build table docs
    for filename, tables_map in all_tables_map.items():
        table_records = await build_table_metadata_store(tables_map, filename)
        table_docs = build_table_documents(table_records)
        documents.extend(table_docs)

    # Build index
    index, nodes = chunk_document(documents)

    hybrid_retriever = hybrid_search(index, nodes)

    cohere_rerank = CohereRerank(
        api_key=COHERE_API_KEY,
        top_n=5,
    )

    from llama_index.core.query_engine import RetrieverQueryEngine

    query_engine = RetrieverQueryEngine.from_args(
        hybrid_retriever,
        llm=llm,
        node_postprocessors=[cohere_rerank],
    )

    print("RAG pipeline ready.")

# ---------------------------
# Upload: save file and rebuild pipeline
# ---------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    print(f"Saved {file.filename}, rebuilding pipeline...")

    # Clear storage cache so chunk_document re-indexes with new file
    import shutil
    storage_path = os.path.join(BASE_DIR, "./storage")
    if os.path.exists(storage_path):
        shutil.rmtree(storage_path)

    await build_pipeline()

    return {"message": "uploaded and indexed", "file": file.filename}


def clean_llm_output(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # remove extra whitespace
    text = text.strip()

    return text

# ---------------------------
# Query endpoint: query the RAG pipeline and return answer + sources + image info
# ---------------------------
@app.post("/query")
async def query_rag(req: QueryRequest):
    if query_engine is None:
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})

    response = await query_engine.aquery(req.query)

    images = []
    has_tables = False
    has_images = False
    sources = []

    for node in response.source_nodes:
        # images
        node_images = node.metadata.get("images", [])
        images.extend(node_images)

        # detect tables and images
        text = node.get_content()
        if "|" in text and "---" in text:
            has_tables = True
        if node_images:
            has_images = True

        # sources
        sources.append({
            "file_name": node.metadata.get("file_name", ""),
            "page": node.metadata.get("page", ""),
            "snippet": node.get_content()[:150],
            "score": round(node.score, 3) if node.score else None,
        })

    images = list(set(images))
    raw_answer = str(response)
    clean_answer = clean_llm_output(raw_answer)

    return JSONResponse(content={
        "answer": clean_answer,
        "images": images,
        "has_tables": has_tables,
        "has_images": has_images,
        "sources": sources,
    })

# ---------------------------
# Health check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}