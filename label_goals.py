#!/usr/bin/env python3
"""
label_goals.py (Component B.2 - v9)

- FINAL SCRIPT (Goals)
- Implements the "Chain-of-Thought" (CoT) prompt to fix "Empty Labels" bug.
- Implements a robust, resumable fetch logic that re-processes all
  recipes but can be safely stopped and restarted.
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
LABEL_VERSION_KEY = "goal_labels" 

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
        logging.FileHandler("labeling_goals_v9.log", mode="a"), # Append mode
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("labeling_goals_v9")

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
# Prompts (v9 - Chain-of-Thought)
# -----------------------------
# The CoT prompt encourages the LLM to reason step-by-step.
USER_PROMPT_TEMPLATE = """
You are a precise Fitness & Diet Coach. Your job is to analyze a recipe and
produce a *single* JSON object.

You MUST follow these steps:
1.  **Analyze**: Silently analyze the {title} and {ingredients}.
2.  **Think**: Reason step-by-step inside a <scratchpad> block.
    - Check for 'weight_loss': (low calorie, low fat, high fiber, high satiety?)
    - Check for 'muscle_gain': (high protein, balanced carbs/fats?)
    - Check for 'low_carb': (low sugar, no grains, no pasta, no potatoes?)
    - Check for 'high_protein': (high in meat, poultry, fish, eggs, tofu, legumes?)
    - Check for 'keto_friendly': (extremely low carb, high fat, no sugar?)
    - Check for 'high_fiber': (high in vegetables, legumes, whole grains?)
3.  **Output**: Based *only* on your reasoning, provide a *single* JSON object
    with the *only* key "goal_labels".

You MUST NOT add any other text outside the JSON.
You MUST provide the <scratchpad> block *before* the JSON.

---
PERFECT EXAMPLE:
---
<scratchpad>
- weight_loss: Yes, this is a 'light vegetable soup', low calorie.
- muscle_gain: No, not high enough in protein.
- low_carb: Yes, 'vegetable soup', no pasta or potatoes listed.
- high_protein: No, ingredients are mostly vegetables.
- keto_friendly: No, contains 'carrots' and 'corn', which are too high in carbs.
- high_fiber: Yes, high in 'vegetables' and 'beans'.
</scratchpad>
{{
  "goal_labels": ["weight_loss", "low_carb", "high_fiber"]
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
# Database Connection
def get_conn():
    log.info("Connecting to Neon database...")
    return psycopg2.connect(NEON_PSYCO_CONN)

# Get Total Recipe Count
def get_total_recipe_count(conn):
    """Gets the total count of recipes."""
    with conn.cursor() as cur:
        query = sql.SQL("SELECT COUNT(*) FROM {};").format(
            sql.Identifier(RECIPE_TABLE)
        )
        cur.execute(query)
        return cur.fetchone()[0]

# Update Recipe Metadata
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
        
        # We merge the new JSON, e.g. {"goal_labels": [...]}
        cur.execute(query, (json.dumps(update_json), recipe_id))

# Parse LLM JSON Output
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
        
        if "goal_labels" not in data:
            log.warning(f"LLM output missing required key 'goal_labels': {llm_output}")
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
    log.info(f"--- Starting Component B.2: Goal Labeling (v9 - Resumable CoT) ---")
    log.info(f"Target key: '{LABEL_VERSION_KEY}'.")
    
    conn = get_conn()
    
    total_to_label = get_total_recipe_count(conn)
    if total_to_label == 0:
        log.warning("No recipes found in table. Exiting.")
        conn.close()
        return

    log.info(f"Found {total_to_label} recipes to label/re-label. Processing...")

    batch_counter = 0
    with tqdm(total=total_to_label, desc="Labeling Goals (v9)") as pbar:
        
        current_offset = 0
        # We must use OFFSET/LIMIT to page through all recipes
        while current_offset < total_to_label:
            
            # --- Connection Reset Logic ---
            if batch_counter > 0 and batch_counter % RESET_CONNECTION_EVERY_N_BATCHES == 0:
                log.info(f"--- Processed {current_offset} recipes. Resetting DB connection... ---")
                conn.commit() 
                conn.close()
                conn = get_conn() 
            # ------------------------------------

            try:
                # --- Fetch ALL recipes in batches using OFFSET/LIMIT ---
                # We are re-processing *all* recipes to apply the CoT prompt.
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    query = sql.SQL("SELECT id, metadata_ FROM {} ORDER BY id LIMIT %s OFFSET %s;").format(
                        sql.Identifier(RECIPE_TABLE)
                    )
                    cur.execute(query, (BATCH_SIZE, current_offset))
                    recipes_to_label = cur.fetchall()
                # ----------------------------------------------------

            except (psycopg2.InterfaceError, psycopg2.OperationalError) as e:
                log.error(f"DB connection failed while fetching batch: {e}. Reconnecting...")
                conn.close()
                conn = get_conn()
                continue # Retry the same batch
                
            if not recipes_to_label:
                log.info("No more recipes to label found.")
                break # Should be finished
            
            recipes_updated = 0
            for row in recipes_to_label:
                recipe_id = row['id']
                current_metadata = row['metadata_']
                
                # --- RESUME LOGIC (Inside Batch) ---
                # Check if this recipe *already* has non-empty, non-error labels
                existing_labels = current_metadata.get(LABEL_VERSION_KEY)
                if isinstance(existing_labels, list) and len(existing_labels) > 0:
                    # This recipe was successfully labeled in a *previous* v9 run
                    # We can skip it.
                    continue 
                # -------------------------------------
                
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
                        if not parsed_data.get("goal_labels"):
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
                current_offset += len(recipes_to_label) # Move to the next page
                
                if len(recipes_to_label) < BATCH_SIZE:
                    break # We are done
                continue # Go to the next batch
            
            # If we 'break'ed from the 'for' loop due to DB error,
            # this 'continue' will restart the 'while' loop.
            # The 'OFFSET' will not have been increased, so we retry the same batch.
            continue 

    conn.close()
    log.info(f"--- ✅ Component B.2 (Goals) v9 is complete. ---")

if __name__ == "__main__":
    main()