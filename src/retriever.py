"""
retriever.py
Handles all retrieval concerns:
  - Chunking documents into nodes
  - Building / loading the vector index
  - Hybrid BM25 + dense retriever with reciprocal reranking
  - Cohere reranker post-processor
  - Assembling the final RetrieverQueryEngine
"""

import os
from dotenv import load_dotenv

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine

from src.utils import VECTOR_DIR, load_retriever_config
from src.generator import llm

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

_cfg = load_retriever_config()


# ─────────────────────────────────────────────────────────────────────────────
# Chunking & indexing
# ─────────────────────────────────────────────────────────────────────────────

def build_index(documents=None):
    """
    Return (index, nodes).

    If *documents* is None and a persisted index exists in VECTOR_DIR,
    the index is loaded from disk. Otherwise a new index is built and persisted.
    """
    storage_path = str(VECTOR_DIR)

    if documents is None and VECTOR_DIR.exists() and any(VECTOR_DIR.iterdir()):
        print("Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
        nodes = list(index.docstore.docs.values())
        print("Done")
        return index, nodes

    if documents is None:
        raise ValueError("No documents provided and no persisted index found.")

    chunk_size = _cfg.get("chunk_size", 256)
    chunk_overlap = _cfg.get("chunk_overlap", 50)

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    print("Chunking documents into nodes...")
    nodes = splitter.get_nodes_from_documents(documents)

    print("Creating new index...")
    index = VectorStoreIndex.from_documents(nodes)
    index.storage_context.persist(storage_path)
    print("Done")

    return index, nodes


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retriever
# ─────────────────────────────────────────────────────────────────────────────

def build_hybrid_retriever(index, nodes):
    """Return a QueryFusionRetriever combining dense and BM25 search."""
    ret_cfg = _cfg.get("retriever", {})
    top_k = ret_cfg.get("similarity_top_k", 10)
    num_queries = ret_cfg.get("num_queries", 1)
    mode = ret_cfg.get("mode", "reciprocal_rerank")

    dense_retriever = index.as_retriever(similarity_top_k=top_k)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)

    return QueryFusionRetriever(
        [dense_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=num_queries,
        mode=mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Query engine assembly
# ─────────────────────────────────────────────────────────────────────────────

def build_query_engine(index, nodes) -> RetrieverQueryEngine:
    """Build the full query engine: hybrid retriever + Cohere reranker."""
    hybrid_retriever = build_hybrid_retriever(index, nodes)

    reranker_cfg = _cfg.get("reranker", {})
    cohere_rerank = CohereRerank(
        api_key=COHERE_API_KEY,
        top_n=reranker_cfg.get("top_n", 5),
    )

    return RetrieverQueryEngine.from_args(
        hybrid_retriever,
        llm=llm,
        node_postprocessors=[cohere_rerank],
    )