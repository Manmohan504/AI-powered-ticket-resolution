import os
import sys
import logging
from tqdm import tqdm

# Ensure we can import from local app dir
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, 'app'))

from app import rag_engine
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
import shutil

# Configure basic logging to console
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)

def manual_ingest():
    print(f"Checking {rag_engine.DATA_RAW_DIR} for documents...")
    
    # Ensure dirs exist
    if not os.path.exists(rag_engine.DATA_RAW_DIR):
        try:
            os.makedirs(rag_engine.DATA_RAW_DIR)
        except: 
            pass
            
    if not os.path.exists(rag_engine.DATA_PROCESSED_DIR):
        os.makedirs(rag_engine.DATA_PROCESSED_DIR, exist_ok=True)

    files = [f for f in os.listdir(rag_engine.DATA_RAW_DIR) if os.path.isfile(os.path.join(rag_engine.DATA_RAW_DIR, f))]
    
    if not files:
        print("No new documents found to ingest.")
        return

    print(f"Found {len(files)} documents. Processing with 3-stage pipeline...")
    print(f"  Embedding model: {rag_engine.EMBEDDING_MODEL}")
    print(f"  Max semantic chunk: {rag_engine.MAX_SEMANTIC_CHUNK_CHARS} chars")
    
    all_chunks = []
    
    # Stage 1 & 2 & 3: Load → Header Split → Semantic Chunk
    for f in tqdm(files, desc="Processing Files"):
        file_path = os.path.join(rag_engine.DATA_RAW_DIR, f)
        try:
            if f.lower().endswith(".pdf"):
                # Stage 1: Layout-aware PDF extraction
                print(f"\n  📄 {f}: Layout-aware extraction...")
                docs = rag_engine._load_pdf_layout_aware(file_path)
                # Stages 2 & 3: Header splitting + Semantic chunking
                chunks = rag_engine.chunk_documents(docs, source_type="pdf")
                all_chunks.extend(chunks)
                print(f"     → {len(chunks)} chunks created")

            elif f.lower().endswith(".txt"):
                print(f"\n  📝 {f}: Loading text...")
                loader = TextLoader(file_path)
                docs = loader.load()
                # Stage 3 only: Semantic chunking
                chunks = rag_engine.chunk_documents(docs, source_type="txt")
                all_chunks.extend(chunks)
                print(f"     → {len(chunks)} chunks created")

        except Exception as e:
            print(f"\n  ❌ Error processing {f}: {e}")

    if not all_chunks:
        print("No valid content extracted.")
        return

    print(f"\nTotal Chunks: {len(all_chunks)}")
    
    # Show chunk statistics
    chunk_sizes = [len(c.page_content) for c in all_chunks]
    print(f"  Avg chunk size: {sum(chunk_sizes)//len(chunk_sizes)} chars")
    print(f"  Min chunk size: {min(chunk_sizes)} chars")
    print(f"  Max chunk size: {max(chunk_sizes)} chars")
    
    # Count chunking methods
    methods = {}
    for c in all_chunks:
        method = c.metadata.get("chunking", "header_only")
        methods[method] = methods.get(method, 0) + 1
    print(f"  Chunking methods: {methods}")
    
    # Embed and Store with nomic-embed-text
    print(f"\nGenerating Embeddings with {rag_engine.EMBEDDING_MODEL}...")
    embeddings = rag_engine.get_embeddings()
    
    # Batch processing for progress bar
    batch_size = 5
    
    if os.path.exists(rag_engine.FAISS_INDEX_PATH):
        try:
            db = FAISS.load_local(rag_engine.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("Loaded existing index.")
        except:
             db = None
    else:
        db = None
    
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding Chunks"):
        batch = all_chunks[i : i + batch_size]
        if db is None:
            db = FAISS.from_documents(batch, embeddings)
        else:
            db.add_documents(batch)
            
    print("Saving Vector Index...")
    db.save_local(rag_engine.FAISS_INDEX_PATH)
    
    # Build BM25 keyword index
    print("Building BM25 keyword index...")
    rag_engine._save_bm25_corpus(all_chunks)
    
    # Move files
    print("Cleaning up...")
    for f in files:
        src = os.path.join(rag_engine.DATA_RAW_DIR, f)
        dst = os.path.join(rag_engine.DATA_PROCESSED_DIR, f)
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f"Failed to move {f}: {e}")

    print("\n✅ Ingestion Complete!")
    print(f"   Chunks indexed: {len(all_chunks)}")
    print(f"   Embedding model: {rag_engine.EMBEDDING_MODEL}")
    print(f"   FAISS index: {rag_engine.FAISS_INDEX_PATH}")
    print(f"   BM25 index: {rag_engine.BM25_INDEX_PATH}")
    print("   You can now run the main app.")

if __name__ == "__main__":
    manual_ingest()