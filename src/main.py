"""
main.py - FastAPI application
Owns the lifespan, pipeline orchestration, and HTTP endpoints (/upload, /query, /health).
"""

import asyncio
import json
import logging

from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llama_index.core.node_parser import SentenceSplitter

from src.ingest import parse_documents, query_table
from src.retriever import build_index, insert_nodes, get_all_nodes, build_query_engine
from src.utils import RAW_DIR, IMAGE_DIR, VECTOR_DIR, clean_llm_output, load_retriever_config, setup_logging
from src.generator import llm

setup_logging()
logger = logging.getLogger(__name__)

cfg = load_retriever_config()

_pipeline_lock = asyncio.Lock()


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
_index = None


async def _build_pipeline() -> None:
    async with _pipeline_lock:
        global _query_engine, _index

        logger.info("Building RAG pipeline")

        docstore_file = VECTOR_DIR / "docstore.json"
        if docstore_file.exists():
            logger.info("Existing docstore found, loading from Qdrant + docstore")
            _index, nodes = build_index()
        else:
            logger.info("No persisted docstore, running full ingestion pipeline")
            documents = await parse_documents()
            logger.info("Ingested %s documents", len(documents))
            _index, nodes = build_index(documents)

        _query_engine = build_query_engine(_index, nodes)
        logger.info("RAG pipeline ready")


class QueryRequest(BaseModel):
    query: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    async with _pipeline_lock:
        global _query_engine, _index

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        file_path = RAW_DIR / file.filename

        logger.info("Received upload: %s", file.filename)
        file_bytes = await file.read()
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        logger.info("Saved %s (%s bytes)", file.filename, len(file_bytes))

        new_documents = await parse_documents(files=[file.filename])
        logger.info("Parsed %s documents from %s", len(new_documents), file.filename)

        splitter = SentenceSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        new_nodes = splitter.get_nodes_from_documents(new_documents)
        logger.info("Generated %s nodes", len(new_nodes))

        if _index is not None:
            insert_nodes(_index, new_nodes)
        else:
            docstore_file = VECTOR_DIR / "docstore.json"
            if docstore_file.exists():
                logger.info("Loading persisted index before incremental insert")
                _index, _ = build_index()
                insert_nodes(_index, new_nodes)
            else:
                logger.warning("No docstore found, building fresh index")
                _index, _ = build_index(new_documents)

        all_nodes = get_all_nodes(_index)
        _query_engine = build_query_engine(_index, all_nodes)
        logger.info("Upload complete for %s; query engine rebuilt with %s nodes", file.filename, len(all_nodes))

    return {"message": "uploaded and indexed", "file": file.filename}


# ─────────────────────────────────────────────────────────────────────────────
# Table resolution helpers
# ─────────────────────────────────────────────────────────────────────────────

_TABLE_RELEVANCE_PROMPT = """\
The user asked: {query}

The following tables were retrieved from the manual. For each table, decide
whether it likely contains information needed to answer the query.

Tables:
{table_list}

Reply with a JSON object only — no prose, no markdown fences:
{{
  "relevant_tables": [
    {{"file_name": "...", "page": <int>, "table_index": <int>}},
    ...
  ]
}}
If no table is relevant, return {{"relevant_tables": []}}."""


async def _resolve_tables(
    query: str,
    table_nodes: list,
) -> list[dict]:
    """
    Ask the LLM which retrieved table_summary nodes are relevant to *query*.
    Returns a list of full structured tables for the relevant ones.
    """
    if not table_nodes:
        return []

    table_list = "\n\n".join(
        f"[{i}] file={n.metadata['file_name']}  page={n.metadata['page']}  "
        f"table_index={n.metadata['table_index']}\n{n.get_content()}"
        for i, n in enumerate(table_nodes)
    )

    prompt = _TABLE_RELEVANCE_PROMPT.format(query=query, table_list=table_list)

    response = await llm.acomplete(prompt)
    raw = response.text.strip()

    try:
        parsed = json.loads(raw)
        relevant = parsed.get("relevant_tables", [])
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON for table relevance: %s", raw[:200])
        relevant = []

    results = []
    for entry in relevant:
        try:
            table_data = query_table(
                file_name=entry["file_name"],
                page=int(entry["page"]),
                table_index=int(entry["table_index"]),
            )
            if table_data["rows"]:
                results.append({
                    "file_name": entry["file_name"],
                    "page": entry["page"],
                    "table_index": entry["table_index"],
                    **table_data,
                })
        except (KeyError, ValueError) as e:
            logger.warning("Could not resolve table entry %s: %s", entry, e)

    logger.info(
        "Table resolution: %s retrieved, %s judged relevant, %s resolved",
        len(table_nodes), len(relevant), len(results),
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Query endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/query")
async def query_rag(req: QueryRequest):
    if _query_engine is None:
        logger.warning("Query attempted before pipeline was ready")
        return JSONResponse(status_code=503, content={"error": "Pipeline not ready"})

    logger.info("Processing query (length=%s)", len(req.query))
    response = await _query_engine.aquery(req.query)

    text_nodes = []
    table_nodes = []
    images: list[str] = []
    sources: list[dict] = []

    for node in response.source_nodes:
        node_type = node.metadata.get("type", "text")
        score = round(node.score, 3) if node.score is not None else None

        if node_type == "table_summary":
            table_nodes.append(node)
        else:
            text_nodes.append(node)
            node_images = node.metadata.get("images", [])
            images.extend(node_images)

        sources.append({
            "type": node_type,
            "file_name": node.metadata.get("file_name", ""),
            "page": node.metadata.get("page", ""),
            "snippet": node.get_content()[:150],
            "score": score,
        })

    # Resolve tables: ask LLM which summaries are relevant, then fetch rows
    resolved_tables = await _resolve_tables(req.query, table_nodes)

    logger.info(
        "Query complete: %s text nodes, %s table nodes, %s resolved tables, %s images",
        len(text_nodes), len(table_nodes), len(resolved_tables), len(set(images)),
    )

    return JSONResponse(content={
        "answer": clean_llm_output(str(response)),
        "images": list(dict.fromkeys(images)),  # dedupe, preserve order
        "tables": resolved_tables,              # full structured rows for frontend rendering
        "sources": sources,
    })


@app.get("/health")
def health():
    return {"status": "ok"}