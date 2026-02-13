#!/usr/bin/env python3
"""
label_health.py (Component B.1 - v9)

- FINAL SCRIPT (Health)
- Implements the "Chain-of-Thought" (CoT) prompt from v8.
- Implements the "Resumable" fetch logic from v7.
- This script can be stopped and restarted without losing progress.
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

# --- This is our target flat key ---
LABEL_VERSION_KEY = "health_labels" 

# Processing batch size
BATCH_SIZE = 50
RESET_CONNECTION_EVERY_N_BATCHES = 20

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("labeling_health_v9.log", mode="a"), # Append mode
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("labeling_health_v9")

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
# Prompts (v8 - Chain-of-Thought)
# -----------------------------

USER_PROMPT_TEMPLATE = """
You are a precise Medical Nutritionist. Your job is to analyze a recipe and
produce a *single* JSON object.

You MUST follow these steps:
1.  **Analyze**: Silently analyze the {title} and {ingredients}.
2.  **Think**: Reason step-by-step inside a <scratchpad> block.
    - Check for 'diabetic_friendly': (low sugar, high fiber, complex carbs?)
    - Check for 'low_sodium': (low salt, no soy sauce, no processed meats?)
    - Check for 'heart_healthy': (low saturated fat, low cholesterol, high omega-3?)
    - Check for 'low_fat': (low oil, lean protein, no heavy cream/cheese?)
    - Check for 'gluten_free': (no wheat, barley, rye, soy sauce?)
    - Check for 'vegetarian': (no meat, poultry, fish?)
    - Check for 'vegan': (no meat, poultry, fish, dairy, eggs, honey?)
3.  **Output**: Based *only* on your reasoning, provide a *single* JSON object
    with the *only* key "health_labels".

You MUST NOT add any other text outside the JSON.
You MUST provide the <scratchpad> block *before* the JSON.

---
PERFECT EXAMPLE:
---
<scratchpad>
- diabetic_friendly: No, contains 'sugar' and 'white flour'.
- low_sodium: Yes, ingredients are all fresh, no salt added.
- heart_healthy: Yes, contains 'salmon' (omega-3) and 'avocado' (healthy fats).
- low_fat: No, 'salmon' and 'avocado' are high fat (healthy, but not low-fat).
- gluten_free: No, 'white flour' is gluten.
- vegetarian: No, contains 'salmon'.
- vegan: No, contains 'salmon'.
</scratchpad>
{{
  "health_labels": ["low_sodium", "heart_healthy"]
}}

---
NOW, ANALYZE THIS RECIPE.
---
Recipe Title:
{title}

Ingredients:
{ingredients}
"""

# -----------------------------
# Helper Functions (Resumable)
# -----------------------------
def get_conn():
    log.info("Connecting to Neon database...")
    return psycopg2.connect(NEON_PSYCO_CONN)

def fetch_recipes_to_label(conn, batch_size):
    """
    Fetches a batch of recipes that are either:
    1. Not labeled yet (no 'health_labels' key)
    2. FAILED labeling last time (have an 'error' key inside 'health_labels')
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

def update_recipe_metadata(conn, recipe_id, update_json):
    """
    Updates the metadata_ column for a single recipe by merging the new key.
    """
    with conn.cursor() as cur:
        query = sql.SQL("""
            UPDATE {}
            SET metadata_ = metadata_ || %s
            WHERE id = %s;
        """).format(sql.Identifier(RECIPE_TABLE))
        
        # We merge the new JSON, e.g. {"health_labels": [...]}
        cur.execute(query, (json.dumps(update_json), recipe_id))

def parse_llm_json_output(llm_output: str) -> dict | None:
    """
    Robustly parses the LLM's string output, which *must* contain JSON.
    It will ignore the <scratchpad> thoughts.
    """
    try:
        # Aggressively find the *last* '{' and '}'
        # This skips the scratchpad and finds the JSON block.
        start = llm_output.rfind('{')
        end = llm_output.rfind('}')
        
        if start == -1 or end == -1:
            log.warning(f"Could not find JSON '{{' or '}}' in output: {llm_output}")
            return None
            
        llm_output = llm_output[start:end+1]
        data = json.loads(llm_output)
        
        if "health_labels" not in data:
            log.warning(f"LLM output missing required key 'health_labels': {llm_output}")
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
    log.info(f"--- Starting Component B.1: Health Labeling (v9 - Resumable CoT) ---")
    log.info(f"Target key: '{LABEL_VERSION_KEY}'.")
    
    conn = get_conn()
    
    total_to_label = get_recipes_to_label_count(conn)
    if total_to_label == 0:
        log.info(f"✅ All recipes are already labeled with '{LABEL_VERSION_KEY}'. Exiting.")
        conn.close()
        return

    log.info(f"Found {total_to_label} recipes remaining to label. Processing...")

    batch_counter = 0
    with tqdm(total=total_to_label, desc="Labeling Health (v9)") as pbar:
        while True:
            # --- Connection Reset Logic ---
            if batch_counter > 0 and batch_counter % RESET_CONNECTION_EVERY_N_BATCHES == 0:
                log.info(f"--- Processed {batch_counter * BATCH_SIZE} recipes. Resetting DB connection... ---")
                conn.commit() 
                conn.close()
                conn = get_conn() 
            # ------------------------------------

            try:
                recipes_to_label = fetch_recipes_to_label(conn, BATCH_SIZE)
            except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
                log.error(f"DB connection failed while fetching batch: {e}. Reconnecting...")
                conn.close()
                conn = get_conn()
                continue
                
            if not recipes_to_label:
                log.info("No more recipes to label found.")
                break
            
            recipes_updated = 0
            for row in recipes_to_label:
                recipe_id = row['id']
                current_metadata = row['metadata_']
                
                try:
                    title = current_metadata.get("title", "")
                    ingredients = current_metadata.get("cleaned_ingredients_str", "")
                    
                    if not title or not ingredients:
                        log.warning(f"Skipping {recipe_id}: missing title or ingredients.")
                        update_json = {LABEL_VERSION_KEY: {"error": "missing_data"}}
                        update_recipe_metadata(conn, recipe_id, update_json)
                        continue

                    user_prompt = USER_PROMPT_TEMPLATE.format(title=title, ingredients=ingredients)
                    
                    llm_output = None
                    for attempt in range(3):
                        try:
                            response = Settings.llm.complete(user_prompt)
                            llm_output = str(response)
                            break
                        except Exception as e:
                            log.warning(f"LLM call failed (attempt {attempt+1}/3): {e}")
                            time.sleep(2)
                    
                    if llm_output is None:
                        log.error(f"LLM failed after 3 attempts for recipe {recipe_id}.")
                        update_json = {LABEL_VERSION_KEY: {"error": "llm_failed_all_retries"}}
                        update_recipe_metadata(conn, recipe_id, update_json)
                        continue 

                    parsed_data = parse_llm_json_output(llm_output)
                    
                    if parsed_data:
                        update_json = parsed_data
                        if not parsed_data.get("health_labels"):
                            log.warning(f"LLM returned an empty list for {recipe_id}.")
                        else:
                            recipes_updated += 1 # Count as a success
                            
                        update_recipe_metadata(conn, recipe_id, update_json)
                    else:
                        update_json = {LABEL_VERSION_KEY: {"error": "failed_to_parse"}}
                        update_recipe_metadata(conn, recipe_id, update_json)
                        log.error(f"Failed to label recipe {recipe_id}. Response: {llm_output[:100]}...")

                except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
                    log.error(f"DB connection failed during update for {recipe_id}: {e}. Reconnecting...")
                    conn.close()
                    conn = get_conn()
                    # CRITICAL: We break inner loop and retry the whole batch
                    # This avoids skipping the recipe that caused the disconnect
                    log.info("Retrying current batch...")
                    break 
                except Exception as e:
                    log.error(f"Critical error processing recipe {recipe_id}: {e}")
                    update_json = {LABEL_VERSION_KEY: {"error": str(e)}}
                    update_recipe_metadata(conn, recipe_id, update_json)
            
            # This 'else' belongs to the 'for' loop
            # It only runs if the 'for' loop completes *without* a 'break'
            else: 
                conn.commit()
                pbar.update(len(recipes_to_label))
                log.info(f"Processed batch. Successfully labeled {recipes_updated}/{len(recipes_to_label)} with non-empty labels.")
                batch_counter += 1
                
                if len(recipes_to_label) < BATCH_SIZE:
                    break # We are done
                continue # Go to the next batch
            
            # If we 'break'ed from the 'for' loop due to DB error,
            # this 'continue' will restart the 'while' loop.
            # The 'fetch_recipes_to_label' will re-fetch the batch.
            continue 

    conn.close()
    log.info(f"--- ✅ Component B.1 (Health) v9 is complete. ---")

if __name__ == "__main__":
    main()