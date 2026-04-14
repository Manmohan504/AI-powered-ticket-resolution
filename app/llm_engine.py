import ollama
import logging
import re

import config
import rag_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------
# qwen2.5:3b — Best-in-class small model for RAG tasks
# 32K context window, excellent instruction following & context comprehension
MODEL_NAME = "qwen2.5:3b"

GENERIC_RESPONSE_PATTERNS = (
    "contact support",
    "reach out to your administrator",
    "not enough information",
    "unable to determine",
    "please provide more details",
    "out of my knowledge base memory",
)

def check_model_availability():
    """Checks if the model is available locally, pulls if not."""
    try:
        # List available models
        models_response = ollama.list()
        
        # Robust parsing for different ollama versions
        model_names = []
        if 'models' in models_response:
            for m in models_response['models']:
                if isinstance(m, dict):
                    model_names.append(m.get('name', ''))
                    model_names.append(m.get('model', '')) # Some versions use 'model'

        # Check against likely variations
        if MODEL_NAME not in model_names and f"{MODEL_NAME}:latest" not in model_names:
            logging.info(f"Model {MODEL_NAME} not found. Pulling...")
            ollama.pull(MODEL_NAME)
            logging.info(f"Model {MODEL_NAME} pulled successfully.")
        else:
            logging.info(f"Model {MODEL_NAME} is ready.")
    except Exception as e:
        logging.warning(f"Error checking model list ({e}). Attempting pull to be safe...")
        try:
            ollama.pull(MODEL_NAME)
        except Exception as pull_error:
            logging.error(f"Failed to pull model: {pull_error}")

def _slugify_filename(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return "_".join(tokens[:6]) if tokens else "missing_knowledge_article"

def _suggest_kb_filename(title, description, category):
    source_text = f"{title} {description}".strip().lower()
    tokens = [
        token for token in re.findall(r"[a-z0-9]+", source_text)
        if token not in {"the", "and", "for", "with", "from", "that", "this", "have", "need", "cannot", "cant"}
    ]
    phrase = " ".join(tokens[:6]) or category or "knowledge gap"
    return f"{_slugify_filename(phrase)}_guide.md"

def _calculate_confidence(retrieval_score, kb_context_found, resolution_text, had_error):
    if had_error:
        return 0.0

    confidence = retrieval_score
    if kb_context_found:
        confidence += 0.2
    if resolution_text and len(resolution_text.strip()) >= 80:
        confidence += 0.1

    lowered = resolution_text.lower()
    if any(pattern in lowered for pattern in GENERIC_RESPONSE_PATTERNS):
        return 0.0

    return max(0.0, min(1.0, round(confidence, 3)))

def analyze_ticket(title, description, priority, category):
    """
    Uses the LLM to generate a resolution and AI quality metadata.
    Optimized prompt structure for qwen2.5:3b with RAG context.
    """
    logging.info("Retrieving relevant context...")
    # Retrieve top-3 chunks (increased from 2 — semantic chunks are smaller & more focused)
    retrieval = rag_engine.get_relevant_context(f"{title} {description}", k=3)
    context = retrieval.get("context_text", "")
    retrieval_score = retrieval.get("retrieval_score", 0.0)
    kb_context_found = retrieval.get("kb_context_found", False)

    # Structured prompt optimized for qwen2.5:3b
    # The model excels with clear system/user role separation and structured instructions
    system_prompt = (
        "You are an IT support resolution engine. "
        "Use ONLY the provided knowledge base context to generate solutions. "
        "If the context does not contain relevant information to answer the ticket, "
        "you MUST reply exactly with: 'I am sorry, but this is out of my knowledge base memory. There is not enough information.' "
        "Do not guess. Be concise. Do not introduce yourself."
    )

    user_prompt = f"""### Knowledge Base Context:
{context if context else "No relevant knowledge base articles found."}

### Support Ticket:
- **Title:** {title}
- **Description:** {description}
- **Priority:** {priority}
- **Category:** {category}

### Task:
Generate a step-by-step resolution for this ticket based on the knowledge base context above.
"""

    if not kb_context_found:
        return {
            "category": category,
            "resolution_text": "I am sorry, but this is out of my knowledge base memory. There is not enough information.",
            "confidence_score": 0.0,
            "resolution_status": "unresolved",
            "retrieval_score": retrieval_score,
            "kb_context_found": False,
            "context_matches": [],
            "suggested_kb_filename": _suggest_kb_filename(title, description, category),
            "error": None,
        }

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        )
        resolution_text = response['message']['content'].strip()
        confidence_score = _calculate_confidence(
            retrieval_score=retrieval_score,
            kb_context_found=kb_context_found,
            resolution_text=resolution_text,
            had_error=False,
        )
        resolved_threshold = config.get_float_env("AI_CONFIDENCE_THRESHOLD", 0.65)
        tentative_threshold = config.get_float_env("AI_TENTATIVE_THRESHOLD", 0.40)
        if confidence_score >= resolved_threshold:
            resolution_status = "resolved"
        elif confidence_score >= tentative_threshold:
            resolution_status = "tentative"
        else:
            resolution_status = "unresolved"
        return {
            "category": category,
            "resolution_text": resolution_text,
            "confidence_score": confidence_score,
            "resolution_status": resolution_status,
            "retrieval_score": retrieval_score,
            "kb_context_found": kb_context_found,
            "context_matches": retrieval.get("matches", []),
            "suggested_kb_filename": (
                None if resolution_status == "resolved"
                else _suggest_kb_filename(title, description, category)
            ),
            "error": None,
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        logging.error(f"LLM Error: {error_msg}")
        return {
            "category": category,
            "resolution_text": f"Failed to generate resolution. Details: {error_msg}",
            "confidence_score": 0.0,
            "resolution_status": "unresolved",
            "retrieval_score": retrieval_score,
            "kb_context_found": kb_context_found,
            "context_matches": retrieval.get("matches", []),
            "suggested_kb_filename": _suggest_kb_filename(title, description, category),
            "error": error_msg,
        }

if __name__ == "__main__":
    # Test run
    analysis = analyze_ticket("Internet down", "My wifi is not connecting", "High", "Network")
    print(f"Category: {analysis['category']}")
    print(f"Resolution: {analysis['resolution_text']}")
