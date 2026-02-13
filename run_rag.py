#!/usr/bin/env python3
"""
run_rag.py (Component C - v5.7 - FINAL FIX)

- v5.6 was a SUCCESS. It fixed the "0 nodes" bug by using
  raw SQL. We retrieved 40 nodes.
-
- THE NEW BUG: The log shows "[ERROR] ... 'str' object has
  no attribute 'partial_format'". This happens when we pass
  the 3 successful nodes to the synthesizer.
-
- THE FIX (v5.7): We must import 'PromptTemplate' and wrap
  our SYNTHESIZER_PROMPT_TEMPLATE string in it. This is
  what the synthesizer expects.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Set
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine 

import psycopg2
import psycopg2.extras

from llama_index.core import (
    Settings,
    VectorStoreIndex, 
)
# --- v5.7 FIX ---
# We must import PromptTemplate
from llama_index.core.prompts import PromptTemplate
# --- END FIX ---
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.schema import NodeWithScore, TextNode

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# -----------------------------
# Configuration (Unchanged)
# -----------------------------
NEON_SQLALCHEMY_CONN_STR = "postgresql+psycopg2://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"
NEON_ASYNC_CONN_STR = "postgresql+asyncpg://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"
NEON_PSYCO_CONN = "postgresql://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"

RECIPE_TABLE = "data_recipes_data" 
OLLAMA_MODEL = "llama3:8b"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384

HEALTH_LABEL_KEY = "health_labels"
GOAL_LABEL_KEY = "goal_labels"

ALLOWED_HEALTH_LABELS: Set[str] = {
    "diabetic_friendly", "low_sodium", "heart_healthy", 
    "low_fat", "gluten_free", "vegetarian", "vegan"
}
ALLOWED_GOAL_LABELS: Set[str] = {
    "weight_loss", "muscle_gain", "low_carb", 
    "high_protein", "keto_friendly", "high_fiber"
}
PRE_FETCH_K = 50

# -----------------------------
# Logging (Unchanged)
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rag_pipeline")

# -----------------------------
# LlamaIndex Settings (Unchanged)
# -----------------------------
log.info(f"Setting up LLM ({OLLAMA_MODEL}) and Embedding Model...")
try:
    Settings.llm = Ollama(model=OLLAMA_MODEL)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    log.info("Models configured successfully.")
except Exception as e:
    log.error(f"Failed to initialize models. Is Ollama running? Error: {e}")
    sys.exit(1)

# -----------------------------
# Prompts & Extractor (Unchanged)
# -----------------------------
EXTRACTOR_PROMPT_TEMPLATE = """
You are a **silent, deterministic JSON-only intent classifier**.
Your task is to analyze the user's request and output a clean JSON structure **without any explanation or natural language text**.

### OUTPUT REQUIREMENTS
- Output **only** a JSON object with exactly two keys: "health_labels" and "goal_labels".
- Both values must be arrays of strings chosen **only** from the predefined lists below.
- If no label applies, return an empty array [] for that key.
- Your output must start with '{{' and end with '}}' (no extra text, comments, or markdown).

### ALLOWED LABELS
- health_labels: ["diabetic_friendly", "low_sodium", "heart_healthy", "low_fat", "gluten_free", "vegetarian", "vegan"]
- goal_labels: ["weight_loss", "muscle_gain", "low_carb", "high_protein", "keto_friendly", "high_fiber"]

### MAPPING RULES
- "overweight", "obese", "slim down", "lose fat", "cutting" -> "weight_loss"
- "gain weight", "build muscle", "bulk", "muscle building", "working out", "gym" -> "muscle_gain" or "high_protein"
- "diabetes", "diabetic", "pre-diabetic", "blood sugar" -> "diabetic_friendly"
- "heart problem", "cardio issue", "high blood pressure", "high cholesterol" -> "heart_healthy" and/or "low_sodium"
- "low carb", "keto" -> "low_carb" or "keto_friendly"
- "vegetarian", "vegan" -> same label respectively
- "gluten free", "celiac" -> "gluten_free"
- "low fat", "reduce fat" -> "low_fat"
- "high fiber", "digestive health" -> "high_fiber"

### EXAMPLES

User Request:
"I’m diabetic and trying to lose weight."

Output:
{{
  "health_labels": ["diabetic_friendly"],
  "goal_labels": ["weight_loss"]
}}

---

User Request:
"I want some vegan meals that help me gain muscle and are high protein."

Output:
{{
  "health_labels": ["vegan"],
  "goal_labels": ["muscle_gain", "high_protein"]
}}

---

NOW, FOLLOW THE SAME PATTERN STRICTLY.
Do not add any reasoning or explanation.
Return only valid JSON for the user’s request below.

User Request:
{query_str}
"""

def parse_llm_json_output(llm_output: str) -> Optional[Dict]:
    """Robustly parses the LLM's JSON output."""
    try:
        start = llm_output.find('{')
        end = llm_output.rfind('}')
        if start == -1 or end == -1:
            return None
        payload = llm_output[start:end+1]
        data = json.loads(payload)
        data["health_labels"] = data.get("health_labels") or []
        data["goal_labels"] = data.get("goal_labels") or []
        return data
        
    except Exception as e:
        log.error(f"Failed to parse LLM output: {e}. Output was: {llm_output}")
        return None

def validate_llm_labels(parsed_json: Dict) -> Dict:
    """
    Sanitizes the LLM's output to ensure only "Allowed" labels
    are kept, preventing hallucinations.
    """
    raw_health = parsed_json.get("health_labels", [])
    raw_goals = parsed_json.get("goal_labels", [])
    
    clean_health = [label for label in raw_health if label in ALLOWED_HEALTH_LABELS]
    clean_goals = [label for label in raw_goals if label in ALLOWED_GOAL_LABELS]
    
    clean_data = {
        "health_labels": clean_health,
        "goal_labels": clean_goals
    }
    
    print("\n" + "="*20 + " DEBUG: Step 1b: Validator " + "="*20)
    print(f"LLM Parsed JSON: {parsed_json}")
    print(f"Sanitized JSON: {clean_data}")
    print("="*60 + "\n")
    
    return clean_data

def extract_user_requirements(query_str: str) -> Optional[Dict]:
    """Uses the LLM to extract structured labels from a user's query."""
    prompt = EXTRACTOR_PROMPT_TEMPLATE.format(query_str=query_str)
    try:
        response = Settings.llm.complete(prompt)
        parsed = parse_llm_json_output(str(response))
        
        # --- DEBUG STEP 1 ---
        print("\n" + "="*20 + " DEBUG: Step 1: Extractor " + "="*20)
        print(f"User Query: '{query_str}'")
        print(f"LLM Raw Output:\n{response}")
        print("="*60 + "\n")
        
        if not parsed:
            return None
            
        validated_data = validate_llm_labels(parsed)
        return validated_data
        
    except Exception as e:
        log.error(f"Error during LLM extraction: {e}")
        return None

# -----------------------------
# --- Manual Filter Function (Unchanged) ---
# -----------------------------
def manual_filter_nodes(nodes: List[NodeWithScore], health_labels: List[str], goal_labels: List[str]) -> List[NodeWithScore]:
    """
    Manually filters a list of retrieved nodes in Python
    to ensure correct JSON array matching.
    """
    filtered_nodes = []
    
    # --- DEBUG STEP 3b: Manual Filter ---
    print("\n" + "="*20 + " DEBUG: Step 3b: Manual Filter " + "="*20)
    print(f"Applying manual filter: Health={health_labels}, Goals={goal_labels}")
    
    for node in nodes:
        meta = node.metadata
        
        node_health = meta.get(HEALTH_LABEL_KEY, [])
        node_goals = meta.get(GOAL_LABEL_KEY, [])
        
        health_ok = all(h in node_health for h in health_labels)
        goal_ok = all(g in node_goals for g in goal_labels)
        
        if health_ok and goal_ok:
            print(f"  [PASS] Node: {meta.get('title', 'No Title')}")
            filtered_nodes.append(node)
        else:
            print(f"  [FAIL] Node: {meta.get('title', 'No Title')}")
            print(f"         Need-H: {health_labels} | Has-H: {node_health}")
            print(f"         Need-G: {goal_labels} | Has-G: {node_goals}")
            
    print(f">>> Result: {len(filtered_nodes)} nodes passed manual filter.")
    print("="*60 + "\n")
    
    return filtered_nodes

# -----------------------------
# Synthesizer prompt
# -----------------------------

# --- v5.7 FIX ---
# We wrap the string in the PromptTemplate class
SYNTHESIZER_PROMPT_TEMPLATE = PromptTemplate("""
You are a helpful and enthusiastic nutrition assistant.
Your job is to answer the user's request using *only* the recipes provided
to you as context.
NEVER make up a recipe.
NEVER use your internal knowledge.
If no recipes are provided, or if the provided recipes do not seem
relevant, you MUST state: "I'm sorry, I couldn't find any recipes
in my database that match all of your specific requirements."
User's request:
{query_str}
Here are the recipes I found that match your request:
---
{context_str}
---
Please present these recipes to the user in a friendly and helpful way.
For each recipe, list its title and briefly mention its ingredients.
""")
# --- END FIX ---

# -----------------------------
# Smarter Retriever logic
# -----------------------------

# --- v5.5 FIX: Raw SQL Query Function (Unchanged) ---
RAW_SQL_QUERY_TEMPLATE = f"""
SELECT id, text, metadata_, (embedding <-> %s) AS distance 
FROM {RECIPE_TABLE}
ORDER BY distance 
LIMIT %s;
"""

def fetch_nodes_raw_sql(query_str: str, top_k: int) -> List[NodeWithScore]:
    """
    Connects via psycopg2 and runs a raw SQL vector query.
    Bypasses the LlamaIndex PGVectorStore object.
    """
    unfiltered_nodes = []
    conn = None
    cur = None 
    try:
        # 1. Get query embedding
        query_embedding = Settings.embed_model.get_query_embedding(query_str)
        # Convert embedding to the string format pgvector expects
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # 2. Connect to DB
        conn = psycopg2.connect(NEON_PSYCO_CONN)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        # 3. Execute raw SQL
        cur.execute(RAW_SQL_QUERY_TEMPLATE, (embedding_str, top_k))
        results = cur.fetchall()

        # 4. Manually reconstruct NodeWithScore objects
        for row in results:
            node = TextNode(
                id_=str(row['id']),
                text=row['text'],
                metadata=row['metadata_']
            )
            score = 1.0 - row['distance'] 
            unfiltered_nodes.append(NodeWithScore(node=node, score=score))

    except Exception as e:
        log.error(f"Error during raw SQL vector query: {e}", exc_info=True)
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            
    return unfiltered_nodes
# --- END v5.5 FIX ---


def retrieve_with_relaxation(query_str: str, extracted_labels: Dict, top_k:int = 3):
    """
    Attempts to retrieve results with progressive relaxation.
    """
    health = list(extracted_labels.get("health_labels") or [])
    goals = list(extracted_labels.get("goal_labels") or [])

    # --- v5.7 FIX ---
    # Pass the PromptTemplate object to the synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=SYNTHESIZER_PROMPT_TEMPLATE, 
    )
    # --- END FIX ---

    # Helper to run a single attempt
    def run_attempt(h_list, g_list):
        log.info(f"--- Attempting retrieval with: Health={h_list}, Goals={g_list} ---")
        
        if not h_list and not g_list:
             # This check is now more important.
             # An empty filter with raw SQL will just return 50 random
             # vector-similar nodes, which isn't what the user wants.
             # We must have at least one label to filter by.
            log.warning("run_attempt called with no filters. Skipping.")
            return None

        # --- DEBUG STEP 3a: Vector Search ---
        print("\n" + "="*20 + " DEBUG: Step 3a: Vector Search (Wide Net) " + "="*20)
        print(f"Running vector search for k={PRE_FETCH_K} (via raw SQL)...")
        
        unfiltered_nodes = fetch_nodes_raw_sql(query_str, PRE_FETCH_K)
            
        print(f">>> Result: {len(unfiltered_nodes)} nodes retrieved (pre-filter).")
        print("="*60 + "\n")
        
        # 2. Manually filter these nodes in Python
        filtered_nodes = manual_filter_nodes(unfiltered_nodes, h_list, g_list)
        
        if not filtered_nodes:
            log.warning("Attempt failed, 0 recipes passed manual filter.")
            return None
        
        # 3. Manually call the synthesizer with our clean list.
        final_nodes = filtered_nodes[:top_k]
        
        try:
            result = response_synthesizer.synthesize(
                query=query_str,
                nodes=final_nodes
            )
            txt = str(result).strip()
            
            # --- DEBUG STEP 4 ---
            print("\n" + "="*20 + " DEBUG: Step 4: Synthesizer " + "="*20)
            print(f"Synthesizer Raw Output:\n{txt}")
            print("="*60 + "\n")
            
            if "I'm sorry, I couldn't find any recipes" in txt or "Empty Response" in txt:
                log.warning("Attempt failed, synthesizer returned empty response.")
                return None
            
            if txt:
                return txt
            return None
        except Exception as e:
            log.error(f"Error during retrieval attempt: {e}", exc_info=True) # Log the full trace
            return None

    # --- RELAXATION STRATEGY (Unchanged) ---
    if health and goals:
        log.info("RELAXATION: Attempt 1: Health + All Goals")
        txt = run_attempt(health, goals)
        if txt:
            return txt, {"strategy":"health+all_goals", "health":health, "goals":goals}
    if health and len(goals) > 1:
        for n in range(len(goals) - 1, 0, -1):
            sub_goals = goals[:n]
            log.info(f"RELAXATION: Attempt 2: Health + Subset Goals ({sub_goals})")
            txt = run_attempt(health, sub_goals)
            if txt:
                return txt, {"strategy":"health+subset_goals", "health":health, "goals":sub_goals}
    if health:
        log.info("RELAXATION: Attempt 3: Health Only")
        txt = run_attempt(health, [])
        if txt:
            return txt, {"strategy":"health_only", "health":health, "goals":[]}
    if goals:
        log.info("RELAXATION: Attempt 4a: Goals All")
        txt = run_attempt([], goals)
        if txt:
            return txt, {"strategy":"goals_all", "health":[], "goals":goals}
        if len(goals) > 1:
            for n in range(len(goals) - 1, 0, -1):
                sub_goals = goals[:n]
                log.info(f"RELAXATION: Attempt 4b: Goals Subset ({sub_goals})")
                txt = run_attempt(health, sub_goals)
                if txt:
                    return txt, {"strategy":"goals_subset", "health":[], "goals":sub_goals}

    log.info("RELAXATION: All strategies failed.")
    return None, {"strategy":"no_matches"}

# -----------------------------
# Main (Unchanged)
# -----------------------------
def main():
    log.info("--- Initializing RAG Pipeline (Component C - v5.7 - FINAL) ---")

    # 1. Connect to the Vector Store (We only do this to check connection)
    try:
        conn = psycopg2.connect(NEON_PSYCO_CONN)
        conn.close()
        log.info("✅ Successfully connected to Neon PGVectorStore (connection test OK).")
    except Exception as e:
        log.error(f"CRITICAL: Failed to connect to database: {e}")
        return

    # interactive loop
    log.info("--- RAG Pipeline is ready. Ask me for recommendations. ---")
    print("Example: 'I am diabetic and want a high_protein, low_carb meal.'")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            query_str = input("Ask for a recommendation: ")
            if query_str.lower() in ["exit", "quit"]:
                log.info("Exiting pipeline.")
                break
            
            print(f"\n[PIPELINE_START] Processing query: '{query_str}'\n")
            extracted = extract_user_requirements(query_str)
            if not extracted:
                print("\nSorry, I had trouble understanding your request.\n")
                continue
            if not extracted.get("health_labels") and not extracted.get("goal_labels"):
                print("\nI'm sorry, I can only provide recommendations for specific labels (like 'vegetarian', 'low_carb', etc.).\n")
                continue

            text, detail = retrieve_with_relaxation(
                query_str=query_str, 
                extracted_labels=extracted, 
                top_k=3
            )

            if text:
                print("\n--- Here's what I found for you ---\n")
                print(text)
                print("\n" + "="*50 + "\n")
                log.info(f"Returned results using strategy: {detail.get('strategy')}")
            else:
                print("\nI'm sorry, I couldn't find any recipes in my database that match all of your specific requirements.\n")
                log.info("No recipes matched; informed user.")

        except Exception as e:
            log.error(f"An error occurred in the query loop: {e}", exc_info=True)
            print(f"\nSorry, an internal error occurred. Please try again.\n")

if __name__ == "__main__":
    main()