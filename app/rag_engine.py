import os
import shutil
import logging
import pickle
from typing import Dict, List

import ollama
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

# Configure Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.getcwd()
if os.path.exists(os.path.join(BASE_DIR, "data")):
    DATA_ROOT = os.path.join(BASE_DIR, "data")
elif os.path.exists(os.path.join(BASE_DIR, "..", "data")):
    DATA_ROOT = os.path.join(BASE_DIR, "..", "data")
else:
    DATA_ROOT = "data"

DATA_RAW_DIR = os.path.join(DATA_ROOT, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
FAISS_INDEX_PATH = os.path.join(DATA_ROOT, "processed", "faiss_index")
BM25_INDEX_PATH = os.path.join(DATA_ROOT, "processed", "bm25_docs.pkl")

# ---------------------------------------------------------------------------
# Model Configuration — tuned to each model's context window
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "nomic-embed-text"   # 8 192-token context, purpose-built for embeddings
# Chunk sizes calibrated for nomic-embed-text (8K tokens ≈ 32K chars)
# We keep chunks well under the limit so each embedding is focused & precise.
MAX_SEMANTIC_CHUNK_CHARS = 2000        # Trigger semantic split above this
FALLBACK_CHUNK_SIZE = 1500             # RecursiveCharacter fallback
FALLBACK_CHUNK_OVERLAP = 200           # Overlap for fallback splitter

# ---------------------------------------------------------------------------
# Search Configuration
# ---------------------------------------------------------------------------
RELEVANCE_THRESHOLD = 0.25            # Minimum similarity to include a result (0-1)
HYBRID_WEIGHT_SEMANTIC = 0.6          # 60% weight to vector/semantic search
HYBRID_WEIGHT_KEYWORD = 0.4           # 40% weight to BM25/keyword search
SEARCH_FETCH_K = 6                    # Fetch extra candidates for re-ranking
SEARCH_FINAL_K = 4                    # Return top-k after filtering & re-ranking

# ---------------------------------------------------------------------------
# Embeddings — using the official langchain-ollama integration
# ---------------------------------------------------------------------------
from langchain_ollama import OllamaEmbeddings as _LCOllamaEmbeddings


def get_embeddings():
    """Returns embeddings instance using nomic-embed-text (purpose-built)."""
    return _LCOllamaEmbeddings(model=EMBEDDING_MODEL)


# Keep a lightweight wrapper so external code that imported OllamaEmbeddings
# from this module (e.g. ingest.py) keeps working.
OllamaEmbeddings = get_embeddings  # callable alias

# ---------------------------------------------------------------------------
# Stage 1 — Layout-Aware PDF Loading (pymupdf4llm → Markdown)
# ---------------------------------------------------------------------------

def _load_pdf_layout_aware(file_path: str) -> List[Document]:
    """
    Extracts structured Markdown from a PDF using pymupdf4llm.
    Preserves headers, tables, lists, and section structure.
    Falls back to basic PyPDFLoader if pymupdf4llm is unavailable.
    """
    try:
        import pymupdf4llm

        md_text = pymupdf4llm.to_markdown(file_path)
        logging.info(f"Layout-aware extraction: {len(md_text)} chars from {os.path.basename(file_path)}")
        return [
            Document(
                page_content=md_text,
                metadata={"source": file_path, "loader": "pymupdf4llm"},
            )
        ]
    except ImportError:
        logging.warning("pymupdf4llm not installed — falling back to PyPDFLoader")
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path).load()
    except Exception as e:
        logging.error(f"Layout-aware extraction failed for {file_path}: {e}")
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(file_path).load()


# ---------------------------------------------------------------------------
# Stage 2 — Header-Based Splitting (MarkdownHeaderTextSplitter)
# ---------------------------------------------------------------------------

HEADERS_TO_SPLIT_ON = [
    ("#", "Header_1"),
    ("##", "Header_2"),
    ("###", "Header_3"),
]


def _split_by_headers(documents: List[Document]) -> List[Document]:
    """
    Splits Markdown documents by header boundaries.
    Each resulting chunk gets header metadata (Header_1, Header_2, etc.).
    Non-Markdown documents (e.g. plain .txt) pass through unchanged.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # Keep headers in chunk text for context
    )

    split_docs: List[Document] = []
    for doc in documents:
        # Only apply header splitting if document looks like Markdown
        if "#" in doc.page_content:
            header_chunks = header_splitter.split_text(doc.page_content)
            for chunk in header_chunks:
                # Merge original metadata with header metadata
                merged_meta = {**doc.metadata, **chunk.metadata}
                split_docs.append(
                    Document(page_content=chunk.page_content, metadata=merged_meta)
                )
        else:
            split_docs.append(doc)

    logging.info(f"Header splitting: {len(documents)} docs → {len(split_docs)} sections")
    return split_docs


# ---------------------------------------------------------------------------
# Stage 3 — Semantic Sub-Chunking (for large sections)
# ---------------------------------------------------------------------------

def _semantic_split(documents: List[Document]) -> List[Document]:
    """
    Applies semantic chunking to sections that exceed MAX_SEMANTIC_CHUNK_CHARS.
    Smaller sections pass through unchanged.

    Uses SemanticChunker with nomic-embed-text to find natural meaning
    boundaries, falling back to RecursiveCharacterTextSplitter if
    semantic chunking is unavailable or fails.
    """
    final_chunks: List[Document] = []
    large_docs: List[Document] = []

    # Separate small (pass-through) and large (need splitting) docs
    for doc in documents:
        if len(doc.page_content) <= MAX_SEMANTIC_CHUNK_CHARS:
            final_chunks.append(doc)
        else:
            large_docs.append(doc)

    if not large_docs:
        logging.info(f"Semantic split: all {len(documents)} chunks are ≤{MAX_SEMANTIC_CHUNK_CHARS} chars — no splitting needed")
        return final_chunks

    # Try semantic chunking first
    semantic_success = False
    try:
        from langchain_experimental.text_splitter import SemanticChunker

        embeddings = get_embeddings()
        semantic_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # Split at large meaning shifts
        )

        for doc in large_docs:
            try:
                sub_chunks = semantic_splitter.split_text(doc.page_content)
                for chunk_text in sub_chunks:
                    final_chunks.append(
                        Document(
                            page_content=chunk_text,
                            metadata={**doc.metadata, "chunking": "semantic"},
                        )
                    )
                semantic_success = True
            except Exception as e:
                logging.warning(f"Semantic split failed for a chunk: {e} — using fallback")
                # Fall through to fallback for this doc
                _fallback_split_doc(doc, final_chunks)

    except ImportError:
        logging.warning("langchain-experimental not installed — using fallback splitter")

    # Fallback: RecursiveCharacterTextSplitter for any remaining large docs
    if not semantic_success:
        for doc in large_docs:
            _fallback_split_doc(doc, final_chunks)

    logging.info(
        f"Semantic split: {len(large_docs)} large sections → "
        f"{len(final_chunks) - len([d for d in final_chunks if d in documents])} sub-chunks "
        f"(total: {len(final_chunks)})"
    )
    return final_chunks


def _fallback_split_doc(doc: Document, output_list: List[Document]):
    """Splits a single large document using RecursiveCharacterTextSplitter."""
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_CHUNK_OVERLAP,
    )
    sub_docs = fallback_splitter.split_documents([doc])
    for sub_doc in sub_docs:
        sub_doc.metadata["chunking"] = "recursive_fallback"
        output_list.append(sub_doc)


# ---------------------------------------------------------------------------
# Full 3-Stage Chunking Pipeline
# ---------------------------------------------------------------------------

def chunk_documents(raw_documents: List[Document], source_type: str = "pdf") -> List[Document]:
    """
    Runs the complete 3-stage chunking pipeline:
      1. Layout-aware extraction (already done for PDFs before calling this)
      2. Header-based splitting (for Markdown/PDF content)
      3. Semantic sub-chunking (for large sections)

    Args:
        raw_documents: Documents loaded from files
        source_type: "pdf" or "txt" — determines which stages to apply

    Returns:
        List of optimally-chunked Document objects
    """
    if not raw_documents:
        return []

    if source_type == "pdf":
        # Stage 2: Split by headers
        header_chunks = _split_by_headers(raw_documents)
        # Stage 3: Semantic sub-chunking
        final_chunks = _semantic_split(header_chunks)
    else:
        # For .txt files: skip header splitting, apply semantic directly
        final_chunks = _semantic_split(raw_documents)

    logging.info(f"Pipeline complete: {len(raw_documents)} inputs → {len(final_chunks)} final chunks")
    return final_chunks


# ---------------------------------------------------------------------------
# Document Ingestion (replaces old ingest_documents)
# ---------------------------------------------------------------------------

def ingest_documents():
    """
    Ingests documents from data/raw using the 3-stage pipeline,
    creates embeddings with nomic-embed-text, and updates the FAISS index.
    Moves processed files to data/processed.
    """
    if not os.path.exists(DATA_RAW_DIR):
        try:
            os.makedirs(DATA_RAW_DIR)
        except OSError:
            pass
    if not os.path.exists(DATA_PROCESSED_DIR):
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    files = [
        f for f in os.listdir(DATA_RAW_DIR)
        if os.path.isfile(os.path.join(DATA_RAW_DIR, f))
    ]

    if not files:
        logging.info("No new documents to ingest.")
        return

    logging.info(f"Found {len(files)} new documents. Starting ingestion...")

    all_chunks: List[Document] = []

    for f in files:
        file_path = os.path.join(DATA_RAW_DIR, f)
        try:
            if f.lower().endswith(".pdf"):
                # Stage 1: Layout-aware PDF extraction
                docs = _load_pdf_layout_aware(file_path)
                # Stages 2 & 3: Header splitting + Semantic chunking
                chunks = chunk_documents(docs, source_type="pdf")
                all_chunks.extend(chunks)

            elif f.lower().endswith(".txt"):
                loader = TextLoader(file_path)
                docs = loader.load()
                # Stage 3 only: Semantic chunking
                chunks = chunk_documents(docs, source_type="txt")
                all_chunks.extend(chunks)

        except Exception as e:
            logging.error(f"Failed to process {f}: {e}")

    if not all_chunks:
        logging.warning("No valid content extracted from any documents.")
        return

    logging.info(f"Total chunks to embed: {len(all_chunks)}")

    # Embed with nomic-embed-text and store in FAISS
    embeddings = get_embeddings()

    if os.path.exists(FAISS_INDEX_PATH):
        try:
            db = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings,
                allow_dangerous_deserialization=True,
            )
            db.add_documents(all_chunks)
            logging.info("Updated existing FAISS index.")
        except Exception as e:
            logging.error(f"Error loading existing index, creating new one: {e}")
            db = FAISS.from_documents(all_chunks, embeddings)
    else:
        db = FAISS.from_documents(all_chunks, embeddings)
        logging.info("Created new FAISS index.")

    db.save_local(FAISS_INDEX_PATH)

    # --- Build BM25 keyword index ---
    logging.info("Building BM25 keyword index...")
    _save_bm25_corpus(all_chunks)

    # Move processed files
    for f in files:
        src = os.path.join(DATA_RAW_DIR, f)
        dst = os.path.join(DATA_PROCESSED_DIR, f)
        try:
            shutil.move(src, dst)
        except Exception as e:
            logging.error(f"Failed to move {f}: {e}")

    logging.info("Ingestion complete.")


# ---------------------------------------------------------------------------
# Similarity Helpers
# ---------------------------------------------------------------------------
import re
import numpy as np
from rank_bm25 import BM25Okapi


def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes cosine similarity between two vectors. Returns value in [-1, 1]."""
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return float(dot_product / (magnitude_a * magnitude_b))


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# BM25 Index Management
# ---------------------------------------------------------------------------

def _save_bm25_corpus(chunks: List[Document]):
    """Saves the chunk texts for BM25 indexing."""
    corpus_data = [
        {"content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(corpus_data, f)
    logging.info(f"BM25 corpus saved: {len(corpus_data)} documents → {BM25_INDEX_PATH}")


def _load_bm25():
    """Loads BM25 corpus and builds the index. Returns (bm25, corpus_data) or (None, [])."""
    if not os.path.exists(BM25_INDEX_PATH):
        return None, []
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            corpus_data = pickle.load(f)
        tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus_data]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, corpus_data
    except Exception as e:
        logging.error(f"Failed to load BM25 index: {e}")
        return None, []


# ---------------------------------------------------------------------------
# Stage A — Hybrid Search (FAISS + BM25) → Top 10
# ---------------------------------------------------------------------------

def _hybrid_search(query: str, fetch_k: int = None) -> List[Dict]:
    """
    Performs hybrid search combining:
      - FAISS vector search (semantic, 60% weight)
      - BM25 keyword search (exact match, 40% weight)
    Returns top fetch_k candidates with combined scores.
    """
    fetch_k = fetch_k or SEARCH_FETCH_K

    embeddings = get_embeddings()
    candidates: Dict[str, Dict] = {}  # key = content hash → candidate

    # --- FAISS Semantic Search ---
    faiss_results = []
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            db = FAISS.load_local(
                FAISS_INDEX_PATH, embeddings,
                allow_dangerous_deserialization=True,
            )
            docs_with_scores = db.similarity_search_with_score(query, k=fetch_k)
            for doc, distance in docs_with_scores:
                l2_sim = max(0.0, min(1.0, 1.0 / (1.0 + float(distance))))
                faiss_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "l2_distance": float(distance),
                    "l2_similarity": l2_sim,
                })
        except Exception as e:
            logging.error(f"FAISS search failed: {e}")

    # Normalize FAISS scores to [0, 1] range
    if faiss_results:
        max_faiss = max(r["l2_similarity"] for r in faiss_results)
        min_faiss = min(r["l2_similarity"] for r in faiss_results)
        score_range = max_faiss - min_faiss if max_faiss != min_faiss else 1.0
        for r in faiss_results:
            norm_score = (r["l2_similarity"] - min_faiss) / score_range
            content_key = hash(r["content"])
            candidates[content_key] = {
                **r,
                "faiss_score": norm_score,
                "bm25_score": 0.0,
            }

    # --- BM25 Keyword Search ---
    bm25, corpus_data = _load_bm25()
    if bm25 is not None:
        query_tokens = _tokenize(query)
        bm25_scores = bm25.get_scores(query_tokens)

        # Get top fetch_k by BM25 score
        top_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

        # Normalize BM25 scores
        max_bm25 = bm25_scores[top_indices[0]] if len(top_indices) > 0 else 1.0
        max_bm25 = max_bm25 if max_bm25 > 0 else 1.0

        for idx in top_indices:
            if bm25_scores[idx] <= 0:
                continue
            doc_data = corpus_data[idx]
            content_key = hash(doc_data["content"])
            norm_bm25 = bm25_scores[idx] / max_bm25

            if content_key in candidates:
                # Document found by both — update BM25 score
                candidates[content_key]["bm25_score"] = norm_bm25
            else:
                # Document found only by BM25
                candidates[content_key] = {
                    "content": doc_data["content"],
                    "metadata": doc_data["metadata"],
                    "l2_distance": 0.0,
                    "l2_similarity": 0.0,
                    "faiss_score": 0.0,
                    "bm25_score": norm_bm25,
                }

    # --- Combine Scores ---
    for key in candidates:
        c = candidates[key]
        c["hybrid_score"] = (
            HYBRID_WEIGHT_SEMANTIC * c["faiss_score"]
            + HYBRID_WEIGHT_KEYWORD * c["bm25_score"]
        )

    # Sort by hybrid score, take top fetch_k
    sorted_candidates = sorted(
        candidates.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )[:fetch_k]

    logging.info(
        f"Hybrid search — FAISS: {len(faiss_results)} hits, "
        f"BM25: {len([c for c in candidates.values() if c['bm25_score'] > 0])} hits, "
        f"Combined: {len(sorted_candidates)} candidates"
    )

    return sorted_candidates


# ---------------------------------------------------------------------------
# Stage B — Re-Ranker (Cosine Similarity) → Top 3
# ---------------------------------------------------------------------------

def _rerank_with_cosine(query: str, candidates: List[Dict], final_k: int = None) -> List[Dict]:
    """
    Re-ranks candidates using cosine similarity between
    query embedding and each candidate's embedding.
    Returns top final_k results.
    """
    final_k = final_k or SEARCH_FINAL_K

    if not candidates:
        return []

    embeddings = get_embeddings()
    query_vector = np.array(embeddings.embed_query(query))

    # Embed each candidate and compute cosine similarity
    for candidate in candidates:
        try:
            doc_vector = np.array(embeddings.embed_query(candidate["content"]))
            candidate["cosine_similarity"] = _cosine_similarity(query_vector, doc_vector)
        except Exception:
            candidate["cosine_similarity"] = 0.0

    # Final score = 50% hybrid + 50% cosine (re-rank boost)
    for c in candidates:
        c["final_score"] = (
            0.5 * c["hybrid_score"]
            + 0.5 * max(0.0, c["cosine_similarity"])
        )

    # Sort by final score, take top final_k
    reranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:final_k]

    # Filter below relevance threshold
    reranked = [r for r in reranked if r["final_score"] >= RELEVANCE_THRESHOLD]

    logging.info(
        f"Re-ranking — {len(candidates)} candidates → {len(reranked)} results "
        f"(threshold: {RELEVANCE_THRESHOLD})"
    )

    for i, r in enumerate(reranked):
        logging.info(
            f"  #{i+1}: hybrid={r['hybrid_score']:.3f}, "
            f"cosine={r['cosine_similarity']:.3f}, "
            f"final={r['final_score']:.3f} "
            f"| {r['content'][:60]}..."
        )

    return reranked


# ---------------------------------------------------------------------------
# Context Retrieval — Full Pipeline
# Query → Hybrid Search (Top 10) → Re-Ranker (Top 5) → LLM
# ---------------------------------------------------------------------------

def get_relevant_context(query: str, k: int = 5) -> Dict:
    """
    Full retrieval pipeline:
      1. Hybrid Search (FAISS + BM25) → Top 10 candidates
      2. Re-Ranker (Cosine Similarity) → Top 3 results
      3. Return context for LLM

    Returns dict with context text, scores, and match details.
    """
    if not os.path.exists(FAISS_INDEX_PATH):
        return {
            "context_text": "",
            "kb_context_found": False,
            "retrieval_score": 0.0,
            "matches": [],
        }

    try:
        # Stage A: Hybrid Search → Top 10
        candidates = _hybrid_search(query, fetch_k=10)

        # Stage B: Re-Ranker → Top 3
        reranked = _rerank_with_cosine(query, candidates, final_k=k)

        # Build results
        matches: List[Dict[str, object]] = []
        for r in reranked:
            matches.append({
                "content": r["content"],
                "metadata": r["metadata"],
                "l2_distance": round(r.get("l2_distance", 0), 4),
                "l2_similarity": round(r.get("l2_similarity", 0), 4),
                "cosine_similarity": round(r.get("cosine_similarity", 0), 4),
                "bm25_score": round(r.get("bm25_score", 0), 4),
                "hybrid_score": round(r.get("hybrid_score", 0), 4),
                "final_score": round(r.get("final_score", 0), 4),
                # Backward compatibility
                "distance": r.get("l2_distance", 0),
                "similarity_score": r.get("final_score", 0),
            })

        context_text = "\n\n".join(m["content"] for m in matches)
        retrieval_score = (
            sum(m["final_score"] for m in matches) / len(matches)
            if matches else 0.0
        )

        logging.info(
            f"Pipeline complete — {len(matches)} results, "
            f"avg retrieval score: {retrieval_score:.3f}"
        )

        return {
            "context_text": context_text,
            "kb_context_found": bool(matches),
            "retrieval_score": round(retrieval_score, 3),
            "matches": matches,
        }
    except Exception as e:
        logging.error(f"Error in retrieval pipeline: {e}")
        return {
            "context_text": "",
            "kb_context_found": False,
            "retrieval_score": 0.0,
            "matches": [],
            "error": str(e),
        }


if __name__ == "__main__":
    ingest_documents()

