import os
from multimodal_loader import  extract_tables

PDF_PATH = "../data/User_Manual_SM2_SE_Camera_CT_Ver.2.5.4_EN.pdf"
IMAGE_DIR = "../extracted_images"

# ── TABLES ────────────────────────────────────────────────────────────────────
print("\n=== TABLE EXTRACTION ===")
table_docs = extract_tables(PDF_PATH)
print(f"Total tables found: {len(table_docs)}")
if table_docs:
    print("--- First table ---")
    print(table_docs[0].text)
    print("--- Metadata ---")
    print(table_docs[0].metadata)
else:
    print("No tables found ")

#Verifying tables are correctly extracted and indexed
print("\n=== VERIFY TABLE IS IN INDEX ===")
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

storage_context = StorageContext.from_defaults(persist_dir="../storage")
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=5)

results = retriever.retrieve("PWR Indicator LED status")
for i, r in enumerate(results):
    print(f"\n-- Result {i+1} (score: {r.score:.3f}) --")
    print(f"Type: {r.metadata.get('type', 'text')}")
    print(r.text[:300])

# ── IMAGES ────────────────────────────────────────────────────────────────────
# print("\n=== IMAGE EXTRACTION ===")
# image_docs = extract_images(PDF_PATH, IMAGE_DIR)
# print(f"Total images found: {len(image_docs)}")
# if image_docs:
#     print("--- First image metadata ---")
#     print(image_docs[0].metadata)
#     print("--- Saved image files ---")
#     for f in os.listdir(IMAGE_DIR)[:5]:  # show first 5
#         print(f"  {f}")
# else:
#     print("No images found")