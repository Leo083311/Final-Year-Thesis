#!/usr/bin/env python3
"""
label_recipes.py (Component B - v6)

- Corrects prompt to include a 'Health and Fitness' persona.
- Bumps label version to 'labels_v2' to force a full re-labeling of all data.
- Implements LLM retries and DB connection resets.
"""

import os
import sys
import time
import json
import logging
import psycopg2
import psycopg2.extras
from psycopg2 import sql
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
NEON_PSYCO_CONN = "postgresql://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"
RECIPE_TABLE = "data_recipes_data"
OLLAMA_MODEL = "llama3:8b"

# --- FIX 2: Bumping the version key to force a re-run ---
# This will re-label all 13,501 recipes under a new, corrected key.
LABEL_VERSION_KEY = "labels_v2"

# Processing batch size
BATCH_SIZE = 50
# Reset connection every 20 batches (1000 recipes)
RESET_CONNECTION_EVERY_N_BATCHES = 20

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("labeling.log", mode="w"), # 'w' to start fresh for v2
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("labeling")

# -----------------------------
# LlamaIndex Settings
# -----------------------------
log.info(f"Setting up LLM: {OLLAMA_MODEL} via Ollama...")
try:
    Settings.llm = Ollama(model=OLLAMA_MODEL)
    log.info("LLM settings configured.")
except Exception as e:
    log.error(f"Failed to initialize Ollama LLM: {e}")
    log.error("Please ensure Ollama is running and the 'llama3:8b' model is pulled.")
    sys.exit(1)

# -----------------------------
# Prompts (FIX 1: Broader Persona)
# -----------------------------

USER_PROMPT_TEMPLATE = """
You are a silent, precise JSON-only **Health and Fitness Classifier**.
Your *only* job is to analyze the recipe at the bottom and return a valid JSON object.
You MUST NOT add any introduction, explanation, or conversational text.
You MUST return *only* the JSON structure.

The JSON object MUST have *only* two keys: "health_labels" and "goal_labels".
- "health_labels" must be an array of strings from this *exact* list:
  ["diabetic_friendly", "low_sodium", "heart_healthy", "low_fat", "gluten_free", "vegetarian", "vegan"]
- "goal_labels" must be an array of strings from this *exact* list:
  ["weight_loss", "muscle_gain", "low_carb", "high_protein", "keto_friendly", "high_fiber"]

If no labels apply, return an empty array for that key.

Here is a PERFECT example of your required output:
{{
  "health_labels": ["low_sodium", "heart_healthy", "gluten_free"],
  "goal_labels": ["weight_loss", "low_carb"]
}}

---
NOW, ANALYZE THIS RECIPE. RETURN ONLY THE JSON.
---
Recipe Title:
{title}

Ingredients:
{ingredients}
"""

# -----------------------------
# Helper Functions
# -----------------------------
def get_conn():
    log.info("Connecting to Neon database...")
    return psycopg2.connect(NEON_PSYCO_CONN)

def fetch_recipes_to_label(conn, batch_size):
    """
    Fetches a batch of recipes that are either:
    1. Not labeled yet (no 'labels_v2' key)
    2. FAILED labeling last time (have an 'error' key inside 'labels_v2')
    """
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        query = sql.SQL("""
            SELECT id, metadata_
            FROM {}
            WHERE 
                (NOT (metadata_ ? %s)) 
                OR 
                (metadata_ -> %s ->> 'error' IS NOT NULL)
            LIMIT %s;
        """).format(sql.Identifier(RECIPE_TABLE))
        
        cur.execute(query, (LABEL_VERSION_KEY, LABEL_VERSION_KEY, batch_size))
        return cur.fetchall()

def get_recipes_to_label_count(conn):
    """Gets the total count of recipes left to label (including failed ones)."""
    with conn.cursor() as cur:
        query = sql.SQL("""
            SELECT COUNT(*)
            FROM {}
            WHERE 
                (NOT (metadata_ ? %s)) 
                OR 
                (metadata_ -> %s ->> 'error' IS NOT NULL);
        """).format(sql.Identifier(RECIPE_TABLE))
        
        cur.execute(query, (LABEL_VERSION_KEY, LABEL_VERSION_KEY))
        return cur.fetchone()[0]

def update_recipe_metadata(conn, recipe_id, new_metadata):
    """Updates the metadata_ column for a single recipe."""
    with conn.cursor() as cur:
        query = sql.SQL("""
            UPDATE {}
            SET metadata_ = %s
            WHERE id = %s;
        """).format(sql.Identifier(RECIPE_TABLE))
        
        cur.execute(query, (psycopg2.extras.Json(new_metadata), recipe_id))

def parse_llm_json_output(llm_output: str) -> dict | None:
    """
    Robustly parses the LLM's string output, which should be JSON.
    """
    try:
        # The LLM might wrap the JSON in markdown (```json ... ```)
        if "```json" in llm_output:
            llm_output = llm_output.split("```json\n", 1)[1].rsplit("```", 1)[0]
        
        # Aggressively find the first '{' and last '}'
        start = llm_output.find('{')
        end = llm_output.rfind('}')
        
        if start == -1 or end == -1:
            log.warning(f"Could not find JSON '{{' or '}}' in output: {llm_output}")
            return None
            
        llm_output = llm_output[start:end+1]

        data = json.loads(llm_output)
        
        # Validate the structure
        if "health_labels" not in data or "goal_labels" not in data:
            log.warning(f"LLM output missing required keys 'health_labels' or 'goal_labels': {llm_output}")
            return None
            
        return data
        
    except json.JSONDecodeError:
        log.error(f"Failed to decode JSON from LLM: {llm_output}")
        return None
    except Exception as e:
        log.error(f"An error occurred during LLM output parsing: {e}")
        return None

# -----------------------------
# Main Labeling Loop
# -----------------------------
def main():
    log.info(f"--- Starting Component B: Recipe Labeling (v6 - Fix Goal Labels) ---")
    log.info(f"Using new label key: '{LABEL_VERSION_KEY}'")
    
    conn = get_conn()
    
    total_to_label = get_recipes_to_label_count(conn)
    if total_to_label == 0:
        log.info("✅ All recipes are already successfully labeled. Exiting.")
        conn.close()
        return

    log.info(f"Found {total_to_label} recipes to label. Processing...")

    batch_counter = 0
    with tqdm(total=total_to_label, desc="Labeling Recipes") as pbar:
        while True:
            # --- Connection Reset Logic ---
            if batch_counter > 0 and batch_counter % RESET_CONNECTION_EVERY_N_BATCHES == 0:
                log.info(f"--- Processed {batch_counter * BATCH_SIZE} recipes. Resetting DB connection... ---")
                conn.commit() # Commit final batch before closing
                conn.close()
                conn = get_conn() # Re-establish
            # ------------------------------------

            # 1. Fetch a batch of recipes to label
            try:
                recipes_to_label = fetch_recipes_to_label(conn, BATCH_SIZE)
            except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
                log.error(f"DB connection failed while fetching batch: {e}. Reconnecting...")
                conn.close()
                conn = get_conn()
                continue # Retry fetching in the next loop iteration
                
            if not recipes_to_label:
                log.info("No more recipes to label found.")
                break # Finished
            
            recipes_updated = 0
            for row in recipes_to_label:
                recipe_id = row['id']
                metadata = row['metadata_'] # This is the full metadata blob
                
                try:
                    # 2. Prepare prompt
                    title = metadata.get("title", "")
                    ingredients = metadata.get("cleaned_ingredients_str", "")
                    
                    if not title or not ingredients:
                        log.warning(f"Skipping {recipe_id}: missing title or ingredients.")
                        metadata[LABEL_VERSION_KEY] = {"error": "missing_data"}
                        update_recipe_metadata(conn, recipe_id, metadata)
                        continue

                    user_prompt = USER_PROMPT_TEMPLATE.format(title=title, ingredients=ingredients)
                    
                    # --- LLM Retry Loop ---
                    llm_output = None
                    for attempt in range(3): # Try 3 times
                        try:
                            response = Settings.llm.complete(user_prompt)
                            llm_output = str(response)
                            break # Success!
                        except Exception as e:
                            log.warning(f"LLM call failed (attempt {attempt+1}/3): {e}")
                            time.sleep(2) # Wait 2 seconds before retrying
                    
                    if llm_output is None:
                        log.error(f"LLM failed after 3 attempts for recipe {recipe_id}.")
                        metadata[LABEL_VERSION_KEY] = {"error": "llm_failed_all_retries"}
                        update_recipe_metadata(conn, recipe_id, metadata)
                        continue # Skip to the next recipe
                    # ---------------------------

                    # 4. Parse and update
                    parsed_data = parse_llm_json_output(llm_output)
                    
                    if parsed_data:
                        # Success! Add the new labels to the metadata
                        # This update is safe and preserves old keys like 'labels_v1'
                        metadata[LABEL_VERSION_KEY] = parsed_data
                        update_recipe_metadata(conn, recipe_id, metadata)
                        recipes_updated += 1
                    else:
                        # Failed to parse.
                        metadata[LABEL_VERSION_KEY] = {"error": "failed_to_parse"}
                        update_recipe_metadata(conn, recipe_id, metadata)
                        log.error(f"Failed to label recipe {recipe_id}. Response: {llm_output[:100]}...")

                except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
                    log.error(f"DB connection failed during update for {recipe_id}: {e}. Reconnecting...")
                    conn.close()
                    conn = get_conn()
                    # We can't save this recipe; it will be retried next run
                except Exception as e:
                    log.error(f"Critical error processing recipe {recipe_id}: {e}")
                    metadata[LABEL_VERSION_KEY] = {"error": str(e)}
                    update_recipe_metadata(conn, recipe_id, metadata)
            
            # Commit the batch of updates
            conn.commit()
            pbar.update(len(recipes_to_label))
            log.info(f"Processed batch. Successfully labeled and updated {recipes_updated}/{len(recipes_to_label)}.")
            batch_counter += 1
            
            if len(recipes_to_label) < BATCH_SIZE:
                break # This was the last batch

    conn.close()
    log.info("--- ✅ All recipes have been processed. Component B (v6) is complete. ---")


if __name__ == "__main__":
    main()