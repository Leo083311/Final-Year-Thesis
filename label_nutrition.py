#!/usr/bin/env python3
"""
label_nutrition.py (v2.1 - Anti-Crash Version)

UPDATES:
1. Added 'get_conn_with_retry': Handles 'No route to host' and DNS errors.
   It waits loop-style until internet comes back.
2. Keeps all previous logic (Temp=0, Sanity Checks).
"""

import sys
import time
import json
import logging
import re
import psycopg2
import psycopg2.extras
from psycopg2 import sql
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from tqdm import tqdm

# -----------------------------
# 1. Configuration
# -----------------------------
NEON_PSYCO_CONN = "postgresql://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"
TABLE_NAME = "data_recipes_data"
OLLAMA_MODEL = "llama3:8b"

BATCH_SIZE = 50
RESET_CONNECTION_EVERY_N_BATCHES = 20

# -----------------------------
# 2. Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("labeling_nutrition_v2.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("label_nutrition")

# -----------------------------
# 3. LLM Setup
# -----------------------------
try:
    Settings.llm = Ollama(
        model=OLLAMA_MODEL, 
        request_timeout=120.0, 
        temperature=0.0
    )
except Exception as e:
    log.error(f"Failed to init Ollama: {e}")
    sys.exit(1)

# -----------------------------
# 4. Prompt
# -----------------------------
EXTRACTION_PROMPT = """
You are a precise Nutrition Data Extractor.
Your goal is to calculate nutritional values for ONE SINGLE SERVING.

INSTRUCTIONS:
1. First, look for the "Servings" or "Yield" in the text.
2. Estimate the TOTAL calories/macros for the ingredients.
3. Divide by the SERVINGS to get per-serving values.
4. Output the JSON object.

REQUIRED JSON FORMAT:
{{
  "calories": 500,
  "protein": 30,
  "fat": 15,
  "carbs": 60
}}

RECIPE TO ANALYZE:
Title: {title}
Ingredients: {ingredients}
Instructions: {instructions}

Output your reasoning briefly, then the JSON:
"""

# -----------------------------
# 5. Robust Database Functions
# -----------------------------

def get_conn_with_retry(max_retries=10):
    """
    Attempts to connect to the DB. If internet is down, it waits and retries.
    Waits up to ~10 minutes total before giving up.
    """
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(NEON_PSYCO_CONN)
            conn.autocommit = False
            return conn
        except Exception as e:
            wait_time = (attempt + 1) * 10 # 10s, 20s, 30s...
            log.warning(f"Connection failed (Attempt {attempt+1}/{max_retries}): {e}")
            log.warning(f"Waiting {wait_time} seconds before retrying...")
            time.sleep(wait_time)
    
    log.error("CRITICAL: Could not connect to database after multiple attempts. Internet might be down.")
    sys.exit(1)

def fetch_unlabeled_batch(conn, limit):
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # Resume Logic: Only get False rows
        query = sql.SQL("""
            SELECT id, metadata_ 
            FROM {}
            WHERE nutrition_labeled = FALSE
            LIMIT %s;
        """).format(sql.Identifier(TABLE_NAME))
        cur.execute(query, (limit,))
        return cur.fetchall()

def get_remaining_count(conn):
    with conn.cursor() as cur:
        query = sql.SQL("SELECT COUNT(*) FROM {} WHERE nutrition_labeled = FALSE;").format(sql.Identifier(TABLE_NAME))
        cur.execute(query)
        return cur.fetchone()[0]

def update_nutrition_data(conn, recipe_id, nutrition_data):
    with conn.cursor() as cur:
        query = sql.SQL("""
            UPDATE {}
            SET 
                calories = %s,
                protein_g = %s,
                fat_g = %s,
                carbs_g = %s,
                nutrition_labeled = TRUE
            WHERE id = %s;
        """).format(sql.Identifier(TABLE_NAME))
        
        cur.execute(query, (
            nutrition_data.get('calories', 0),
            nutrition_data.get('protein', 0),
            nutrition_data.get('fat', 0),
            nutrition_data.get('carbs', 0),
            recipe_id
        ))

def mark_as_failed(conn, recipe_id):
    with conn.cursor() as cur:
        query = sql.SQL("""
            UPDATE {} SET nutrition_labeled = TRUE WHERE id = %s;
        """).format(sql.Identifier(TABLE_NAME))
        cur.execute(query, (recipe_id,))

# -----------------------------
# 6. Logic
# -----------------------------

def parse_json_response(response_text):
    try:
        blocks = re.findall(r"\{[\s\S]*?\}", response_text)
        if not blocks: return None
        json_str = blocks[-1]
        json_str = re.sub(r",\s*}", "}", json_str) # Fix trailing commas
        data = json.loads(json_str)
        return {
            "calories": int(data.get("calories", 0)),
            "protein": int(data.get("protein", 0)),
            "fat": int(data.get("fat", 0)),
            "carbs": int(data.get("carbs", 0))
        }
    except Exception:
        return None

def is_sane(data):
    if not data: return False
    c, p, f, cb = data['calories'], data['protein'], data['fat'], data['carbs']
    if any(x < 0 for x in [c, p, f, cb]): return False
    if c > 4000: return False 
    if p > 400 or f > 400 or cb > 600: return False
    return True

# -----------------------------
# 7. Main Loop
# -----------------------------
def main():
    log.info("--- Starting Task 1: Nutrition Extraction (v2.1 Anti-Crash) ---")
    conn = get_conn_with_retry()
    
    total_remaining = get_remaining_count(conn)
    log.info(f"Found {total_remaining} recipes waiting for nutrition labels.")
    
    if total_remaining == 0:
        log.info("All done! Exiting.")
        conn.close()
        return

    batch_counter = 0
    pbar = tqdm(total=total_remaining, desc="Extracting Macros")

    while True:
        # Connection Hygiene
        if batch_counter > 0 and batch_counter % RESET_CONNECTION_EVERY_N_BATCHES == 0:
            conn.commit()
            conn.close()
            conn = get_conn_with_retry() # Use the Safe Retry here
        
        try:
            batch = fetch_unlabeled_batch(conn, BATCH_SIZE)
            if not batch:
                break

            for row in batch:
                r_id = row['id']
                meta = row['metadata_']
                
                title = meta.get('title', 'Unknown')
                ingredients = meta.get('cleaned_ingredients_str') or "No ingredients listed"
                instr_text = meta.get('instructions', '')
                instructions = " ".join(instr_text.split()[:120]) 
                
                prompt = EXTRACTION_PROMPT.format(
                    title=title, ingredients=ingredients, instructions=instructions
                )

                valid_data = None
                for _ in range(2): 
                    try:
                        resp = Settings.llm.complete(prompt)
                        parsed = parse_json_response(str(resp))
                        if is_sane(parsed):
                            valid_data = parsed
                            break
                    except:
                        time.sleep(1)
                
                if valid_data:
                    update_nutrition_data(conn, r_id, valid_data)
                else:
                    log.warning(f"Skipping {r_id}: Failed/Insane Data")
                    mark_as_failed(conn, r_id)

            conn.commit()
            pbar.update(len(batch))
            batch_counter += 1

        except Exception as e:
            # THIS IS THE CRITICAL FIX
            log.error(f"Batch connection lost: {e}")
            log.info("Attempting to reconnect safely...")
            try:
                conn = get_conn_with_retry() # Will try for 10 minutes
                log.info("Reconnected! Resuming...")
            except Exception as fatal:
                log.error("Internet is gone for good. Saving progress and exiting.")
                break

    pbar.close()
    if conn: conn.close()
    log.info("--- Nutrition Extraction Complete ---")

if __name__ == "__main__":
    main()