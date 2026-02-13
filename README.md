This thesis presents a privacy-first, clinically safe AI recommendation engine for personalized nutrition and fitness planning. To solve the critical issue of Large Language Model (LLM) hallucinations in healthcare, the project introduces a novel Two-Tier Architecture that strictly separates sensitive user data (processed locally) from semantic reasoning (processed in the cloud via Google Gemini).

Driven by an Agentic Orchestrator built with LangGraph, the system features a self-correcting "Auto-Repair" loop. It uses Hybrid Retrieval (Semantic Vector Search + SQL) to audit AI-generated meal and workout plans in real-time against strict medical constraints (such as Chronic Kidney Disease or severe allergies). If a safety violation is detected, the agent autonomously corrects it before reaching the user. The resulting system bridges the gap between the intelligence of generative AI and the zero-tolerance safety requirements of digital health, achieving a 100% safety compliance score during testing on complex, multi-morbid profiles.

Key Technical Highlights:
Architecture: Two-Tier / Stateless Cloud Inference
Core Logic: Agentic Workflows (LangGraph) & "Auto-Repair" State Machines
Data Pipeline: Local Llama-3 (8B) for offline data enrichment of 13,000+ recipes
Retrieval: Hybrid RAG using pgvector (Neon DB) and strict SQL filtering
