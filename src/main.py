import os
import sys
import json
import asyncio
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8570ef9 (Add image extraction and downloading)
import re
import fitz  # PyMuPDF


from ast import literal_eval
<<<<<<< HEAD
=======
>>>>>>> b12a9e2 (Update full pipeline with hybrid search and reranking)
=======
>>>>>>> 8570ef9 (Add image extraction and downloading)
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

<<<<<<< HEAD
<<<<<<< HEAD
IMAGE_DIR = "../parsed_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

client = AsyncLlamaCloud(api_key=os.getenv("LLAMA_CLOUD_API_KEY"))
=======
=======
IMAGE_DIR = "../parsed_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


>>>>>>> 8570ef9 (Add image extraction and downloading)
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
>>>>>>> b12a9e2 (Update full pipeline with hybrid search and reranking)

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


#Download and save images
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image_name = f"page_{page_index+1}_img_{img_index}.png"
            image_path = os.path.join(IMAGE_DIR, image_name)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            print(f"Saved {image_name}")

def get_images_for_page(page_number):
    return [
        f for f in os.listdir(IMAGE_DIR)
        if f.startswith(f"page_{page_number}_")
    ]

async def download_images(result):
    """Download images using presigned URLs from images_content_metadata."""
    if not result.images_content_metadata:
        print("  No image metadata found in result.")
        return

    import httpx
    async with httpx.AsyncClient() as http:
        for img_meta in result.images_content_metadata:
            # skip non-image metadata items (e.g. total_count)
            if not hasattr(img_meta, 'filename') or not hasattr(img_meta, 'presigned_url'):
                continue
            
            img_name = img_meta.filename
            url = img_meta.presigned_url

            if not img_name or not url:
                continue

            dest = os.path.join(IMAGE_DIR, img_name)
            if os.path.exists(dest):
                continue
            try:
                print(f"  Downloading {img_name}...")
                response = await http.get(url)
                response.raise_for_status()
                with open(dest, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"  Warning: could not download {img_name}: {e}")

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

            # warn about any images referenced but missing on disk
            for page in pages:
                pass  # images handled by metadata + PDF extraction already

            for page in pages:
                documents.append(Document(text=page["text"], metadata=page["metadata"]))
            continue

        # parse normally
        file_path = os.path.join(data_dir, filename)
        extract_images_from_pdf(file_path)

        print(f"Uploading {filename} to LlamaCloud...")
        file_obj = await llama_cloud_client.files.create(
            file=file_path,
            purpose="parse"
        )

        print("Parsing file...")
        result = await llama_cloud_client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",  
            version="latest",
            agentic_options={
                "custom_prompt": "This is an equipment manual..."
            },
            output_options={
                "markdown": {
                    "tables": {"output_tables_as_markdown": True},
                },
                "images_to_save": ["embedded", "screenshot"],  
            },
            expand=["markdown", "images_content_metadata"]
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

        # download images before saving cache
        await download_images(result)

        with open(cache_file, "w") as f:
            json.dump(pages_to_save, f)

        # build documents from saved pages
        for page in pages_to_save:
            documents.append(
            Document(
                text=page["text"],
                metadata={
                    "file_name": filename,
                    "page": page["metadata"]["page"],
                    "images": get_images_for_page(page["metadata"]["page"])  # 🔥 ADD
                }
                )
            )

    return documents

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======

>>>>>>> 8570ef9 (Add image extraction and downloading)
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

>>>>>>> b12a9e2 (Update full pipeline with hybrid search and reranking)

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
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 8570ef9 (Add image extraction and downloading)

    images = []

    for node in response.source_nodes:
        imgs = node.metadata.get("images", [])
        images.extend(imgs)

    images = list(set(images))

    print("Answer:", response)
    print("Relevant images:", images)
<<<<<<< HEAD
    print(response)
=======
=======
>>>>>>> 8570ef9 (Add image extraction and downloading)
    print(llm_response.response)
>>>>>>> b12a9e2 (Update full pipeline with hybrid search and reranking)

if __name__ == "__main__":
    asyncio.run(main())