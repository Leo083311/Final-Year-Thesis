# 🧬 Personalized Nutrition & Fitness Recommender (Agentic RAG)

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/LangGraph-Agentic_Workflow-orange.svg)](https://python.langchain.com/docs/langgraph)
[![Database](https://img.shields.io/badge/pgvector-Neon_Serverless-green.svg)](https://neon.tech/)
[![LLM](https://img.shields.io/badge/Models-Gemini_1.5_%7C_Llama--3-purple.svg)]()

> **An advanced, privacy-preserving health recommendation engine utilizing a Two-Tier Distributed Architecture, Agentic Orchestration (LangGraph), and Hybrid Retrieval.**

## 📖 Overview
This repository contains the source code for my final-year Bachelor of Engineering in Artificial Intelligence (Honours) at Xiamen University Malaysia thesis (Awarded 4.0 as result). The system automates hyper-personalized nutrition and fitness planning by bridging large language models with deterministic medical constraints. 

To ensure strict data privacy and high-level reasoning, the architecture is split into a **Local Processing Unit** (for secure, offline data enrichment) and a **Stateless Cloud Inference Unit** (for complex semantic logic with zero PII transfer).

## ✨ Key Architectural Features

* **🤖 Agentic Orchestrator & Auto-Repair Failsafe:** Built using **LangGraph**, the system manages complex state transitions. It features a continuous self-correction loop (Reflexion) that dynamically audits and repairs LLM hallucinations before outputting to the user.
* **🔍 Hybrid Retrieval Engine:** Combines semantic vector search with deterministic SQL filtering using **pgvector** and **Neon DB**. This enforces strict medical constraints, successfully blocking conflicting dietary keywords (e.g., filtering high-protein items for Chronic Kidney Disease parameters).
* **🛡️ Zero-PII Privacy Architecture:** Leverages a Two-Tier system. User constraints are sanitized locally, while high-level reasoning is handled via a stateless JSON exchange protocol using **Google Gemini 1.5**. 
* **⚡ Automated Data Enrichment Pipeline:** Utilizes a local **Llama-3 (8B)** model via Ollama to autonomously analyze, label, and index over 13,000 raw culinary and fitness data points offline prior to database ingestion.

## 🛠️ Tech Stack

| Category | Technologies Used |
| :--- | :--- |
| **LLMs & Frameworks** | Google Gemini 1.5, Meta Llama-3 (8B), LangGraph, Prompt Engineering (CoT) |
| **Data & Retrieval** | PostgreSQL (Neon Serverless), pgvector (HNSW Indexing), Hybrid Search |
| **Backend & Compute** | Python (Multiprocessing, ProcessPoolExecutor), RESTful APIs |
| **Data Science** | Pandas, NumPy, Scikit-learn (K-Means, PCA, DBSCAN, Isolation Forest) |

## 🚀 Installation & Setup

### Prerequisites
* Python 3.9+
* [Ollama](https://ollama.com/) (for running Llama-3 locally)
* A [Neon DB](https://neon.tech/) account (for PostgreSQL/pgvector)
* Google Gemini API Key

### 1. Clone the Repository
```bash
git clone [https://github.com/Leo083311/Final-Year-Thesis.git](https://github.com/Leo083311/Final-Year-Thesis.git)
cd Final-Year-Thesis

📬 Contact & Author
Leonard Tye Zi Yang

AI Engineering Undergraduate @ Xiamen University Malaysia

LinkedIn: linkedin.com/in/leonardtyeziyang

Email: leot3108@gmail.com
