from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

app = FastAPI()

# ---------------------------
# Paths (IMPORTANT: make absolute-safe)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "../parsed_images")

# Serve images
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")


# ---------------------------
# Request model
# ---------------------------
class QueryRequest(BaseModel):
    query: str


# ---------------------------
# Global query engine placeholder
# ---------------------------
query_engine = None


# ---------------------------
# Startup: build RAG pipeline here
# ---------------------------
@app.on_event("startup")
async def startup_event():
    global query_engine

    from llama_index.core import VectorStoreIndex
    from llama_index.core.node_parser import SentenceSplitter
    from main import parse_documents_with_llamaparse    
    from main import llm
    documents = await parse_documents_with_llamaparse("../data")

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex(nodes)

    query_engine = index.as_query_engine(
        similarity_top_k=5,
        llm=llm
    )


# ---------------------------
# Query endpoint
# ---------------------------
@app.post("/query")
async def query_rag(req: QueryRequest):
    response = await query_engine.aquery(req.query)

    images = []

    for node in response.source_nodes:
        node_images = node.metadata.get("images", [])
        images.extend(node_images)

    # remove duplicates
    images = list(set(images))

    return JSONResponse(content={
        "answer": str(response),
        "images": images
    })


# ---------------------------
# Health check (optional but useful)
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}