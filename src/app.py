from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import re


app = FastAPI()

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

    from llama_index.core import VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.postprocessor.cohere_rerank import CohereRerank
    from main import parse_documents_with_llamaparse, llm, hybrid_search, chunk_document

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")

    documents = await parse_documents_with_llamaparse(DATA_DIR)

    index, nodes = chunk_document(documents)

    hybrid_retriever = hybrid_search(index, nodes)

    cohere_rerank = CohereRerank(api_key=COHERE_API_KEY, top_n=5)

    from llama_index.core.query_engine import RetrieverQueryEngine
    query_engine = RetrieverQueryEngine.from_args(
        hybrid_retriever,
        llm=llm,
        node_postprocessors=[cohere_rerank],
    )

    print("RAG pipeline ready.")


# ---------------------------
# Startup
# ---------------------------
@app.on_event("startup")
async def startup_event():
    await build_pipeline()


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