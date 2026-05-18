from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi import Request
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
async def query_rag(req: QueryRequest, request: Request):
    if query_engine is None:
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})

    response = await query_engine.aquery(req.query)
    base_url = str(request.base_url).rstrip("/")

    tables = []
    sources = []
    matched_pages = set()  # ← collect pages from text nodes

    for node in response.source_nodes:
        node_type = node.metadata.get("type")

        if node_type == "table":
            markdown = node.metadata.get("table_markdown")
            if markdown:
                tables.append(markdown)

        else:
            # text node — record the page number
            page = node.metadata.get("page")
            if page is not None:
                matched_pages.add(int(page))

            sources.append({
                "file_name": node.metadata.get("file_name", ""),
                "page": page,
                "snippet": node.get_content()[:150],
                "score": round(node.score, 3) if node.score else None,
            })

    # Find all images whose filename page number is in matched_pages
    images = []
    if os.path.exists(IMAGE_DIR):
        for img_name in os.listdir(IMAGE_DIR):
            if not img_name.endswith(".png"):
                continue
            # filename format: page_22_img_0.png → extract 22
            try:
                page_num = int(img_name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if page_num in matched_pages:
                images.append(f"{base_url}/images/{img_name}")

    images = sorted(set(images))
    tables = list(dict.fromkeys(tables))
    raw_answer = str(response)
    clean_answer = clean_llm_output(raw_answer)

    return JSONResponse(content={
        "answer": clean_answer,
        "images": images,
        "tables": tables,
        "has_images": len(images) > 0,
        "has_tables": len(tables) > 0,
        "sources": sources,
    })

# ---------------------------
# Health check
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}