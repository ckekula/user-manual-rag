import os
import sys
import json
import asyncio
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

# parse PDF
async def parse_documents_with_llamaparse(data_dir: str):
    documents = []

    for filename in os.listdir(data_dir):
        if not filename.endswith(".pdf"):
            continue

        cache_file = os.path.join(CACHE_DIR, f"{filename}.json")

        # load from cache
        if os.path.exists(cache_file):
            print(f"Loading cached parse for {filename}...")
            with open(cache_file, "r") as f:
                pages = json.load(f)

            for page in pages:
                documents.append(
                    Document(
                        text=page["text"],
                        metadata=page["metadata"]
                    )
                )
            continue

        # parse normally
        file_path = os.path.join(data_dir, filename)

        print(f"Uploading {filename} to LlamaCloud...")
        file_obj = await llama_cloud_client.files.create(
            file=file_path,
            purpose="parse"
        )

        print("Parsing file...")
        result = await llama_cloud_client.parsing.parse(
            file_id=file_obj.id,
            tier="cost_effective",
            version="latest",
            agentic_options={
                "custom_prompt": "This is an equipment manual..."
            },
            output_options={
                "markdown": {
                    "tables": {"output_tables_as_markdown": True},
                },
                "images_to_save": ["embedded"],
            },
            expand=["markdown"]
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

            documents.append(
                Document(
                    text=page.markdown,
                    metadata={
                        "file_name": filename,
                        "page": page.page_number
                    }
                )
            )

        with open(cache_file, "w") as f:
            json.dump(pages_to_save, f)

    return documents

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
    # Build retrievers
    dense_retriever = index.as_retriever(similarity_top_k=10)

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=10,
    )

    # Hybrid retriever
    hybrid_retriever = QueryFusionRetriever(
        [dense_retriever, bm25_retriever],
        similarity_top_k=10,
        num_queries=1,
        mode="reciprocal_rerank",
    )

    return hybrid_retriever


async def main():
    documents = await parse_documents_with_llamaparse("../data")

    index, nodes = chunk_document(documents)

    hybrid_retriever = hybrid_search(index, nodes)

    # Cohere reranker — returns top 5 after reranking the 20 candidates
    cohere_rerank = CohereRerank(
        api_key=COHERE_API_KEY,
        top_n=5,
    )

    # Query engine with hybrid retrieval + reranking
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
    print(llm_response.response)

if __name__ == "__main__":
    asyncio.run(main())