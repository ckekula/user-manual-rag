import os
import pdfplumber
# import pymupdf4llm
from pathlib import Path
from llama_index.core import Document


def extract_tables(pdf_path: str) -> list[Document]:
    """Extract all tables using pdfplumber — converts each to markdown."""
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for i, table in enumerate(page.extract_tables()):
                if not table:
                    continue
                rows = [r for r in table if any(c for c in r)]
                if len(rows) < 2:
                    continue
                header = "| " + " | ".join(str(c or "") for c in rows[0]) + " |"
                sep    = "| " + " | ".join("---" for _ in rows[0]) + " |"
                body   = ["| " + " | ".join(str(c or "") for c in r) + " |" for r in rows[1:]]
                docs.append(Document(
                    text="\n".join([header, sep] + body),
                    metadata={"source": pdf_path, "page": page_num, "type": "table"}
                ))
    return docs


def extract_images(pdf_path: str, image_dir: str = "extracted_images") -> list[Document]:
    """Extract embedded images using pymupdf4llm — saves to disk, returns metadata docs."""
    os.makedirs(image_dir, exist_ok=True)
    pages = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=True,
        image_path=image_dir,
        image_format="png",
        page_chunks=True,
    )
    docs = []
    for page in pages:
        for img in page.get("images", []):
            img_path = img.get("path") or img.get("name", "")
            if img_path:
                docs.append(Document(
                    text=f"[Image on page {page['metadata'].get('page', '?')}]",
                    metadata={"source": pdf_path, "page": page["metadata"].get("page"),
                              "image_path": img_path, "type": "image"}
                ))
    return docs




def load_all_pdfs(data_dir: str = "../data", image_dir: str = "../extracted_images") -> list[Document]:
    all_docs = []
    for pdf in Path(data_dir).glob("*.pdf"):
        table_docs = extract_tables(str(pdf))
        image_docs = extract_images(str(pdf), image_dir)
        print(f"{pdf.name}: {len(text_docs)} text | {len(table_docs)} tables | {len(image_docs)} images")
        all_docs.extend(text_docs + table_docs + image_docs)
    return all_docs