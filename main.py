import os
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
# setup Arize Phoenix for logging/observability
import phoenix as px
px.launch_app()
llama_index.set_global_handler("arize_phoenix")
from llama_index.query_pipeline import QueryPipeline
from llama_index.prompts import PromptTemplate


load_dotenv()

llm = GoogleGenAI(model="gemini-2.5-flash-lite")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader("./data").load_data()

# chunking
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=50
)

print("Chunking documents into nodes...")
nodes = splitter.get_nodes_from_documents(documents)

# vector index
if os.path.exists("./storage"):
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context)
else:
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    index.storage_context.persist("./storage")

# basic retriever
retriever = index.as_retriever(similarity_top_k=5)

print("Querying the index...")
query_engine = RetrieverQueryEngine.from_args(
    retriever,
    llm=llm
)

print("Generating response...")
response = query_engine.query(
    "List out the features of this device"
)

print(response)