#!/usr/bin/env python3
"""
label_workouts.py (Component B.3)

- Objective: Analyze ingested workouts and infer their 'target_goals'.
- Source Data: Reads 'title', 'type', 'target_muscle' from DB.
- Output: Updates metadata_ with a new "target_goals" list.
  (e.g., ["muscle_gain", "strength"] or ["weight_loss", "cardio"])
"""

import sys
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
NEON_DB_URL = "postgresql://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"
TABLE_NAME = "data_workout_data"
OLLAMA_MODEL = "llama3:8b"

# Processing batch size
BATCH_SIZE = 50

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
log = logging.getLogger("label_workouts")

# -----------------------------
# LlamaIndex Settings
# -----------------------------
try:
    Settings.llm = Ollama(model=OLLAMA_MODEL, request_timeout=120.0)
except Exception as e:
    log.error(f"Failed to init Ollama: {e}")
    sys.exit(1)

# -----------------------------
# The Expert Prompt
# -----------------------------
PROMPT_TEMPLATE = """
You are an expert Fitness Coach. Analyze this exercise and identify its primary goals.

Exercise: "{title}"
Type: "{type}"
Target Muscle: "{target_muscle}"
Level: "{intensity}"

Return a JSON object with a single key "target_goals".
The value must be a list of strings chosen ONLY from:
["weight_loss", "muscle_gain", "strength", "endurance", "flexibility", "stress_relief"]

Rules:
- "Strength" or "Powerlifting" types usually -> ["muscle_gain", "strength"]
- "Cardio" or "Plyometrics" usually -> ["weight_loss", "endurance"]
- "Yoga" or "Stretching" usually -> ["flexibility", "stress_relief"]
- Use your knowledge of the specific exercise title to decide.

Example Output:
{{
  "target_goals": ["muscle_gain", "strength"]
}}

Your JSON response:
"""

def get_conn():
    return psycopg2.connect(NEON_DB_URL)

def fetch_unlabeled_workouts(conn):
    """Fetches workouts that don't have 'target_goals' yet."""
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # We check if the 'target_goals' key is missing from the metadata_ JSON
        query = sql.SQL("""
            SELECT id, text, metadata_
            FROM {}
            WHERE NOT (metadata_ ? 'target_goals')
            LIMIT %s;
        """).format(sql.Identifier(TABLE_NAME))
        cur.execute(query, (BATCH_SIZE,))
        return cur.fetchall()

def update_workout(conn, workout_id, new_goals):
    """Merges the new goals into the existing metadata."""
    with conn.cursor() as cur:
        # This SQL updates the JSONB column by merging the new key-value pair
        update_query = sql.SQL("""
            UPDATE {}
            SET metadata_ = metadata_ || %s
            WHERE id = %s;
        """).format(sql.Identifier(TABLE_NAME))
        
        json_payload = json.dumps({"target_goals": new_goals})
        cur.execute(update_query, (json_payload, workout_id))

def parse_json(response_text):
    try:
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0: return None
        return json.loads(response_text[start:end])
    except:
        return None

def main():
    log.info("--- Starting Workout Labeling (Target Goals) ---")
    conn = get_conn()
    
    # Check total remaining
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE NOT (metadata_ ? 'target_goals');")
        remaining = cur.fetchone()[0]
    
    log.info(f"Found {remaining} workouts to label.")
    
    if remaining == 0:
        log.info("All workouts are labeled! Exiting.")
        return

    # Progress bar for the loop
    pbar = tqdm(total=remaining)
    
    while True:
        batch = fetch_unlabeled_workouts(conn)
        if not batch:
            break
            
        for row in batch:
            w_id = row['id']
            meta = row['metadata_']
            
            # Prepare prompt
            prompt = PROMPT_TEMPLATE.format(
                title=meta.get('title', ''),
                type=meta.get('type', ''),
                target_muscle=meta.get('target_muscle', ''),
                intensity=meta.get('intensity', '')
            )
            
            try:
                # Call Local LLM
                response = Settings.llm.complete(prompt)
                data = parse_json(str(response))
                
                if data and "target_goals" in data:
                    goals = data["target_goals"]
                    update_workout(conn, w_id, goals)
                else:
                    log.warning(f"Failed to label {meta.get('title')}")
                    # Mark as failed to avoid infinite loop (optional, or just skip)
                    
            except Exception as e:
                log.error(f"LLM Error: {e}")
        
        conn.commit()
        pbar.update(len(batch))

    pbar.close()
    conn.close()
    log.info("--- Labeling Complete ---")

if __name__ == "__main__":
    main()