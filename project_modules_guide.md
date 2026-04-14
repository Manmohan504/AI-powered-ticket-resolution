# 📦 AI-Based Ticket Resolution System — Project Guide

> An IT support ticket system where users submit issues and a locally-running AI (via Ollama) automatically generates resolutions using **RAG (Retrieval-Augmented Generation)**.

---

## 🧩 Module-by-Module Breakdown

### 1. [database.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/database.py) — The Data Layer

- Manages the **SQLite database** ([support_tickets.db](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/support_tickets.db))
- Creates two tables: [tickets](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/ticket_service.py#28-36) and [users](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/auth_service.py#33-43)
- Provides raw DB functions: [init_db()](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/database.py#13-41), [create_user()](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/database.py#42-55), [get_user()](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/database.py#56-66)
- Every other module that needs DB access calls through this file

> **Think of it as:** The foundation. Nothing works without this.

---

### 2. [auth_service.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/auth_service.py) — User Authentication

- Sits **on top of** [database.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/database.py)
- Hashes passwords using **bcrypt** (never stores plain text)
- Provides [register_user()](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/auth_service.py#13-22), [login_user()](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/auth_service.py#23-32), [create_default_users()](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/auth_service.py#33-43)
- Automatically seeds a default `admin` (password: `admin123`) and `testuser` (password: `user123`) on first launch

> **Think of it as:** The gatekeeper — who can log in and who can't.

---

### 3. [rag_engine.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/rag_engine.py) — Knowledge Base (RAG)

- Implements **Retrieval-Augmented Generation** using a FAISS vector store
- Reads `.pdf` and [.txt](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/ai%20powered.txt) files from `data/raw/`
- Splits documents into 1000-character chunks, generates **embeddings** using the `tinyllama` model via Ollama
- Saves a **FAISS vector index** to `data/processed/faiss_index/`
- On query: searches the index and returns the **top-2 most relevant chunks** as context

> **Think of it as:** The AI's reference library — it searches your documents to find relevant answers.

---

### 4. [llm_engine.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/llm_engine.py) — AI Resolution Generator

- Uses the **`llama3.2:1b`** model via Ollama
- When a ticket arrives, it:
  1. Calls `rag_engine.get_relevant_context()` to retrieve relevant document chunks
  2. Builds a prompt combining the ticket details + retrieved context
  3. Sends to Ollama and returns a bullet-point resolution
- Auto-checks/pulls the model if not locally present

> **Think of it as:** The brain — takes the ticket + context and produces the solution.

---

### 5. [ticket_service.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/ticket_service.py) — Ticket Business Logic

- The **orchestration layer** — connects DB + LLM + RAG
- [submit_ticket()](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/ticket_service.py#6-27) — calls `llm_engine.analyze_ticket()` then saves to DB
- [get_user_tickets()](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/ticket_service.py#37-47) — fetches all tickets for a specific user (history view)
- [get_all_tickets()](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/ticket_service.py#28-36) — fetches all tickets (for admin view)
- [initialize_system()](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/ticket_service.py#59-64) — called on app startup; inits DB and pulls LLM model

> **Think of it as:** The coordinator — it wires everything together.

---

### 6. [app.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/app.py) — The Frontend (Streamlit UI)

- The **user-facing web interface** built with Streamlit
- Has two states: **Login/Signup page** and **Main app** (post-auth)
- Main app has two tabs:
  - 🔥 **New Incident** — form to submit a ticket; shows AI resolution instantly
  - 📂 **My History** — expandable list of past tickets with AI resolutions
- Custom dark-mode CSS styling (Inter font, blue accents `#3b82f6`)

> **Think of it as:** What the end user sees and interacts with.

---

### 7. [ingest.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/ingest.py) — Document Ingestion Script *(root level)*

- A **standalone CLI script** — run manually, not auto-triggered
- Loads PDFs/TXTs from `data/raw/`, embeds them via `rag_engine`, saves FAISS index
- Shows progress bars via `tqdm`
- After ingestion, moves files from `data/raw/` → `data/processed/`

> **Think of it as:** The "setup" script you run once to load your knowledge base.

---

### 8. [fix_db.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/fix_db.py) — Database Migration Utility

- A one-time repair/migration script
- Checks if `priority` and `user_id` columns exist in the [tickets](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/ticket_service.py#28-36) table and adds them if missing
- Used when upgrading an older version of the DB schema

> **Think of it as:** A DB patch tool — not part of the normal runtime flow.

---

### 9. [run.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/run.py) — App Launcher *(root level)*

- Simple utility: runs `streamlit run app/app.py` via Python subprocess
- Sets `PYTHONPATH` so all local imports resolve correctly

> **Think of it as:** The start button.

---

## 🏗️ Build Order (Replicating From Scratch)

Follow this exact order — each step depends on what came before:

| Step | File | Why This Order |
|:----:|------|----------------|
| **1** | [database.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/database.py) | Everything else depends on the DB. Build tables, connection, and CRUD first. |
| **2** | [auth_service.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/auth_service.py) | Needs [database.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/database.py). Implement bcrypt hashing + login/register logic. |
| **3** | [rag_engine.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/rag_engine.py) | Independent of DB. Set up Ollama embeddings, FAISS, and document loading. |
| **4** | [ingest.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/ingest.py) | Needs `rag_engine`. Run this to load knowledge documents and build the vector index. |
| **5** | [llm_engine.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/llm_engine.py) | Needs `rag_engine`. Build the prompt + Ollama LLM call that produces resolutions. |
| **6** | [ticket_service.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/ticket_service.py) | Needs `database`, `llm_engine`, `rag_engine`. Orchestrate the full ticket flow. |
| **7** | [app.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/app.py) | Needs `ticket_service` + `auth_service`. Build the UI last — it's the top of the stack. |
| **8** | [run.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/run.py) | Optional launcher wrapper. Add once the app is fully working. |
| **9** | [fix_db.py](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/app/fix_db.py) | Only needed if migrating an older schema. Skip for a fresh build. |

---

## 🔑 Key Dependencies

Install via pip:

```bash
pip install streamlit ollama langchain-community langchain-text-splitters faiss-cpu bcrypt pandas pypdf tqdm
```

---

## ⚠️ Prerequisites

> [!IMPORTANT]
> You must have **Ollama installed and running locally** before starting the app.
> Pull both required models beforehand:
> ```bash
> ollama pull tinyllama      # Used for embeddings in rag_engine.py
> ollama pull llama3.2:1b    # Used for resolution generation in llm_engine.py
> ```

---

## 🏃 Running the App

1. Place your `.pdf` or [.txt](file:///c:/Users/ChetanAtla/Desktop/backup/use/AI-Based-Ticket-Resolution-System-main/ai%20powered.txt) knowledge documents in `data/raw/`
2. Run the ingestion script from the project root:
   ```bash
   python ingest.py
   ```
3. Start the app:
   ```bash
   python run.py
   ```
   Or directly:
   ```bash
   streamlit run app/app.py
   ```

---

## 🗂️ Data Flow Summary

```
User submits ticket (app.py)
        ↓
ticket_service.submit_ticket()
        ↓
llm_engine.analyze_ticket()
    ↓               ↓
rag_engine          ollama (llama3.2:1b)
(FAISS search)      (generates resolution)
        ↓
database.py (saves ticket + resolution)
        ↓
User sees AI resolution (app.py)
```
