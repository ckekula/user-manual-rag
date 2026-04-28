import os
import sys
import json
import asyncio
from ast import literal_eval
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    PromptTemplate,
    load_index_from_storage,
    set_global_handler
)
from llama_index.llms.vllm import Vllm
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings
from llama_cloud import AsyncLlamaCloud

# logging
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index.core").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
set_global_handler("simple")

load_dotenv()
CACHE_DIR = "../parsed_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

client = AsyncLlamaCloud(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))

# initialize the LLM and embedding model
if os.getenv("APP_ENV") == "dev":
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = GoogleGenAI(model="gemini-2.5-flash-lite")
elif os.getenv("APP_ENV") == "prod":
    embed_model = HuggingFaceEmbedding(model_name="Qwen/Qwen3-Embedding-0.6B")
    llm = Vllm(
        model="Qwen3.5-122B-A10B-GGUF",
        tensor_parallel_size=4,
        max_new_tokens=100,
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
        file_obj = await client.files.create(
            file=file_path,
            purpose="parse"
        )

        print("Parsing file...")
        result = await client.parsing.parse(
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

async def main():
    documents = await parse_documents_with_llamaparse("../data")

    print("Chunking documents into nodes...")
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    # index
    if os.path.exists("../storage"):
        print("Loading index from storage...")
        storage_context = StorageContext.from_defaults(persist_dir="../storage")
        index = load_index_from_storage(storage_context)
    else:
        print("Creating new index...")
        index = VectorStoreIndex(nodes)
        index.storage_context.persist("../storage")

    retriever = index.as_retriever(similarity_top_k=5)

    print("Querying the index...")
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        llm=llm
    )

    print("Generating response...")
    response = await query_engine.aquery(
        "What is the name of this device?"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())