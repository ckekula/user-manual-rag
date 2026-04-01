import os
import sys
import asyncio
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    set_global_handler
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.settings import Settings

# logging
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("llama_index.core").setLevel(logging.WARNING)
logging.getLogger("fsspec").setLevel(logging.WARNING)
set_global_handler("simple")

load_dotenv()

# initialize the LLM and embedding model
llm = GoogleGenAI(model="gemini-2.5-flash-lite")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("../data").load_data()

# chunk the document and store the nodes in the vector index
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

print("Chunking documents into nodes...")
nodes = splitter.get_nodes_from_documents(documents)

# vector index
if os.path.exists("../storage"):
    print("Loading index from storage...")
    storage_context = StorageContext.from_defaults(persist_dir="../storage")
    index = load_index_from_storage(storage_context)
else:
    print("Creating new index...")
    index = VectorStoreIndex.from_documents(nodes)
    index.storage_context.persist("../storage")

# basic retriever
retriever = index.as_retriever(similarity_top_k=5)

print("Querying the index...")
query_engine = RetrieverQueryEngine.from_args(
    retriever,
    llm=llm
)

async def main():
    print("Generating response...")
    response = await query_engine.aquery(
        "What is the name of this device?"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(main())