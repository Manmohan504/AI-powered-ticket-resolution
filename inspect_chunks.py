"""Inspect chunks stored in the FAISS vector index."""
import os
import sys
import pickle

sys.path.append(os.path.join(os.getcwd(), "app"))
from app import rag_engine

from langchain_community.vectorstores import FAISS

print("Loading FAISS index...")
embeddings = rag_engine.get_embeddings()
db = FAISS.load_local(rag_engine.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# Get all documents from the docstore
docs = list(db.docstore._dict.values())

print(f"\n{'='*60}")
print(f"Total chunks in index: {len(docs)}")
print(f"{'='*60}\n")

# Statistics
sizes = [len(d.page_content) for d in docs]
print(f"Avg chunk size: {sum(sizes)//len(sizes)} chars")
print(f"Min chunk size: {min(sizes)} chars")
print(f"Max chunk size: {max(sizes)} chars")

# Chunking methods
methods = {}
for d in docs:
    m = d.metadata.get("chunking", "header_only")
    methods[m] = methods.get(m, 0) + 1
print(f"Chunking methods: {methods}")

# Sources
sources = {}
for d in docs:
    src = os.path.basename(d.metadata.get("source", "unknown"))
    sources[src] = sources.get(src, 0) + 1
print(f"Sources: {sources}")

print(f"\n{'='*60}")
print("CHUNK PREVIEW (first 10 chunks)")
print(f"{'='*60}\n")

for i, doc in enumerate(docs[:10]):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    method = doc.metadata.get("chunking", "header_only")
    header = doc.metadata.get("Header_1", "")
    content_preview = doc.page_content
    
    print(f"--- Chunk {i+1} ---")
    print(f"  Source:  {source}")
    print(f"  Method:  {method}")
    if header:
        print(f"  Header:  {header}")
    print(f"  Size:    {len(doc.page_content)} chars")
    print(f"  Preview: {content_preview}...")
    print()

# Ask if user wants to see all
input_text = input("Show ALL chunks? (y/n): ").strip().lower()
if input_text == "y":
    for i, doc in enumerate(docs):
        print(f"\n{'='*60}")
        print(f"CHUNK {i+1}/{len(docs)}")
        print(f"Metadata: {doc.metadata}")
        print(f"Size: {len(doc.page_content)} chars")
        print(f"{'='*60}")
        print(doc.page_content)
