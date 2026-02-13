# 🧬 Personalized Nutrition & Fitness Recommender (Agentic RAG)

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/LangGraph-Agentic_State_Machine-orange.svg)](https://python.langchain.com/docs/langgraph)
[![Database](https://img.shields.io/badge/PostgreSQL-Neon_DB_%7C_pgvector-green.svg)](https://neon.tech/)
[![LLMs](https://img.shields.io/badge/Gemini_2.5_Flash-Llama_3_(8B)-purple.svg)]()

> **A privacy-first, fault-tolerant recommendation engine for complex medical comorbidities, built on a Two-Tier Architecture using LangGraph, Google Gemini, and local Llama-3.**

## 📖 Overview

Standard fitness applications fail for users with complex medical conditions (e.g., Chronic Kidney Disease + Hypertension) because they prioritize simple goal-matching over clinical safety. 

This project solves that by orchestrating an **Agentic RAG** workflow. To satisfy both strict data privacy (zero PII leakage) and high-level reasoning, the system is decoupled into a **Local Processing Unit** (for secure, offline data enrichment and deterministic filtering) and a **Stateless Cloud Inference Unit** (for semantic logic).

## 🧠 Core Architecture



### 1. The Agentic Orchestrator (LangGraph & Gemini)
The core logic is not a linear script, but a state-driven graph built with **LangGraph**. The workflow maintains a strict `AgentState` dictionary that tracks BMR targets, contraindications, and the evolving recommendation plan.

* **Safety Auditor Node:** Before presenting any data, Gemini 2.5 Flash acts as an independent auditor. It inspects the generated 7-day meal plan against the user's specific medical profile.
* **Autonomous Auto-Repair:** If the Auditor detects a clinical risk (e.g., a high-phosphorus meal for a CKD patient), the graph routes to an Auto-Repair node. This node surgically removes the flagged item and triggers a targeted SQL re-query to find a safe substitute, ensuring zero user intervention is required for error recovery.

### 2. Hybrid Retrieval Engine (SQL + pgvector)
Relying purely on semantic vector search is dangerous for medical use cases—you cannot "fuzzily" avoid a peanut allergy.

* **Deterministic Hard Constraints:** I implemented strict SQL `WHERE` and `ILIKE ANY(...)` clauses using `psycopg2`. For instance, if a user has Hypertension, the SQL layer actively blocks keywords like `'%bacon%'` or `'%soy sauce%'` at the database level.
* **Semantic Search:** Only recipes that pass the deterministic medical filters are vectorized and ranked using a 384-dimensional `sentence-transformers/all-MiniLM-L6-v2` model and **HNSW indexing** in Neon PostgreSQL.



### 3. Offline LLM Enrichment Pipeline (ETL)
To build the knowledge base without incurring massive API costs or exposing data, I built an offline data pipeline utilizing a local **Llama-3 (8B)** model via Ollama.

* **Structured Chain-of-Thought (CoT):** Wrote custom CoT prompts to force the local LLM to reason step-by-step before classifying 13,000+ raw culinary recipes into strict JSON metadata schemas (e.g., `health_labels`, `goal_labels`).
* **Fault-Tolerant Ingestion:** Engineered the ingestion scripts to handle network instability and database timeouts. Implemented connection retry loops (`get_conn_with_retry`), batch DB commits, and an MD5 hashing system (`doc_hash`) to allow the pipeline to resume safely if interrupted.
* **Parallel Processing:** Utilized Python's `ProcessPoolExecutor` to parallelize the embedding generation across multiple CPU cores, drastically reducing the time required to vectorize the dataset.

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **State Machine / Orchestration** | LangGraph, Python 3.10 |
| **Cloud LLM (Reasoning/Auditing)** | Google Gemini 2.5 Flash API |
| **Local LLM (Data Enrichment)** | Meta Llama-3 (8B) via Ollama |
| **Vector & Relational Database** | PostgreSQL (Neon Serverless), `pgvector`, `psycopg2` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Concurrency & ETL** | `ProcessPoolExecutor`, `tqdm`, Hash-based Resume State |

## 🚀 Installation & Setup

### Prerequisites
* Python 3.10+
* [Ollama](https://ollama.com/) (Must pull Llama-3: `ollama run llama3:8b`)
* A [Neon DB](https://neon.tech/) account
* Google Gemini API Key

### 1. Clone & Install
```bash
git clone [(https://github.com/Leo083311/Final-Year-Thesis.git)]
cd Agentic-Health-Recommender
pip install -r requirements.txt
