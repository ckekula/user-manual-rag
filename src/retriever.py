"""
retriever.py
Handles all retrieval concerns:
  - Chunking documents into nodes
  - Building / loading the vector index backed by Qdrant
  - Hybrid BM25 + dense retriever with reciprocal reranking
  - Cohere reranker post-processor
  - Assembling the final RetrieverQueryEngine
"""

import os
import logging
from dotenv import load_dotenv

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
import qdrant_client

from src.utils import VECTOR_DIR, load_retriever_config
from src.generator import llm

load_dotenv()

logger = logging.getLogger(__name__)

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")          # None for local
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")

_cfg = load_retriever_config()


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant client (module-level singleton)
# ─────────────────────────────────────────────────────────────────────────────

def _make_qdrant_client() -> qdrant_client.QdrantClient:
    if QDRANT_API_KEY:
        return qdrant_client.QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return qdrant_client.QdrantClient(url=QDRANT_URL)


# ─────────────────────────────────────────────────────────────────────────────
# Chunking & indexing
# ─────────────────────────────────────────────────────────────────────────────

def build_index(documents=None):
    """
    Return (index, nodes).

    Qdrant is used as the vector store; a local SimpleDocumentStore persisted
    under VECTOR_DIR is used to keep node payloads for BM25.

    Behaviour:
      - documents=None  → load existing index (Qdrant collection + docstore).
      - documents=[...] → chunk, embed, and insert into Qdrant; persist docstore.

    Raises ValueError if documents=None and no persisted docstore exists yet.
    """
    docstore_path = str(VECTOR_DIR)

    qclient = _make_qdrant_client()
    vector_store = QdrantVectorStore(
        client=qclient,
        collection_name=QDRANT_COLLECTION,
    )

    # ── Load path ────────────────────────────────────────────────────────────
    if documents is None:
        docstore_file = VECTOR_DIR / "docstore.json"
        if not docstore_file.exists():
            raise ValueError(
                "No documents provided and no persisted docstore found. "
                "Upload at least one PDF first."
            )
        logger.info("Loading index from Qdrant + local docstore")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=docstore_path,
        )
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
        # Retrieve nodes from docstore for BM25
        nodes = list(storage_context.docstore.docs.values())
        logger.info("Loaded %s nodes from docstore", len(nodes))
        return index, nodes

    # ── Build path ───────────────────────────────────────────────────────────
    chunk_size = _cfg.get("chunk_size", 256)
    chunk_overlap = _cfg.get("chunk_overlap", 50)
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    logger.info(
        "Chunking %s documents into nodes (chunk_size=%s, chunk_overlap=%s)",
        len(documents),
        chunk_size,
        chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info("Created %s nodes", len(nodes))

    logger.info("Building Qdrant-backed index in collection %s", QDRANT_COLLECTION)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        # Docstore keeps node text/metadata locally for BM25
        docstore=SimpleDocumentStore(),
    )
    storage_context.docstore.add_documents(nodes)

    # from_nodes avoids a second chunking pass that from_documents would trigger
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    # Persist docstore (Qdrant persists itself)
    storage_context.persist(docstore_path)
    logger.info("Index built and docstore persisted to %s", docstore_path)

    return index, nodes


def insert_nodes(index: VectorStoreIndex, new_nodes: list, persist: bool = True) -> None:
    """
    Insert *new_nodes* into an existing index and update the local docstore.

    Separating insertion from index construction keeps main.py's upload
    handler simple and ensures the docstore stays consistent with Qdrant.
    """

    existing_ids = set(index.storage_context.docstore.docs.keys())
    deduped = [n for n in new_nodes if n.node_id not in existing_ids]

    if not deduped:
        logger.info("All %s nodes already indexed, skipping insert", len(new_nodes))
        return

    logger.info("Inserting %s new nodes (%s provided)", len(deduped), len(new_nodes))

    for node in deduped:
        index.insert_nodes([node])
        
    # Add to Qdrant
    for node in new_nodes:
        index.insert_nodes([node])

    # Add to docstore so subsequent BM25 builds see the new nodes
    index.storage_context.docstore.add_documents(new_nodes)

    if persist:
        index.storage_context.persist(str(VECTOR_DIR))
        logger.info("Persisted %s new nodes to docstore", len(new_nodes))


def get_all_nodes(index: VectorStoreIndex) -> list:
    """Return all nodes currently stored in the index's docstore."""
    return list(index.storage_context.docstore.docs.values())


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retriever
# ─────────────────────────────────────────────────────────────────────────────

def build_hybrid_retriever(index: VectorStoreIndex, nodes: list) -> QueryFusionRetriever:
    """Return a QueryFusionRetriever combining dense (Qdrant) and BM25 search."""
    ret_cfg = _cfg.get("retriever", {})
    top_k = ret_cfg.get("similarity_top_k", 10)
    num_queries = ret_cfg.get("num_queries", 1)
    mode = ret_cfg.get("mode", "reciprocal_rerank")

    logger.info(
        "Building hybrid retriever (top_k=%s, num_queries=%s, mode=%s, nodes=%s)",
        top_k,
        num_queries,
        mode,
        len(nodes),
    )

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

def build_query_engine(index: VectorStoreIndex, nodes: list) -> RetrieverQueryEngine:
    """Build the full query engine: hybrid retriever + Cohere reranker."""
    hybrid_retriever = build_hybrid_retriever(index, nodes)

    reranker_cfg = _cfg.get("reranker", {})
    cohere_rerank = CohereRerank(
        api_key=COHERE_API_KEY,
        top_n=reranker_cfg.get("top_n", 5),
    )

    logger.info("Building query engine with Cohere reranker top_n=%s", reranker_cfg.get("top_n", 5))

    return RetrieverQueryEngine.from_args(
        hybrid_retriever,
        llm=llm,
        node_postprocessors=[cohere_rerank],
    )