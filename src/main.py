import os
import sys
import json
import asyncio
import re
import fitz  # PyMuPDF


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

IMAGE_DIR = "../parsed_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

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
        file_obj = await client.files.create(
            file=file_path,
            purpose="parse"
        )

        print("Parsing file...")
        result = await client.parsing.parse(
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

    images = []

    for node in response.source_nodes:
        imgs = node.metadata.get("images", [])
        images.extend(imgs)

    images = list(set(images))

    print("Answer:", response)
    print("Relevant images:", images)
    print(response)

if __name__ == "__main__":
    asyncio.run(main())