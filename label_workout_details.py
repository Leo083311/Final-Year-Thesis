#!/usr/bin/env python3
"""
label_workout_details.py (Task 2 - V3.0 Enhanced Science Mode)

OBJECTIVE:
Extract bio-mechanical metadata ("Compound", "Push", "Tier 1") to enable
advanced planning logic (Bro Splits, PPL, Strength Periodization).

FEATURES:
- Classifies exercises by Mechanic (Compound/Isolation)
- Classifies by Force (Push/Pull)
- Estimates Duration & Calorie Burn
- v2.1 Anti-Crash Logic
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
TABLE_NAME = "data_workout_data"
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
        logging.FileHandler("labeling_workouts_science.log", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("label_workouts")

# -----------------------------
# 3. LLM Setup (Deterministic)
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
# 4. The "Science-Based" Prompt
# -----------------------------
EXTRACTION_PROMPT = """
You are an expert Strength & Conditioning Coach (CSCS).
Analyze the biomechanics of the exercise below for a professional periodization plan.

REQUIRED DATA:
1. "mechanic": strictly "compound" (multi-joint) or "isolation" (single-joint) or "isometric".
2. "force_type": strictly "push" (away from center), "pull" (towards center), or "static".
3. "tier": "primary" (Main lifts: Squat, Bench, Deadlift, OHP, etc.) or "secondary" (Accessory/Cardio).
4. "duration_min": Estimated time to complete 3 standard sets (including rest).
5. "cal_per_min": Average calorie burn per minute.

REQUIRED JSON FORMAT:
{{
  "mechanic": "compound",
  "force_type": "push",
  "tier": "primary",
  "duration_min": 12,
  "cal_per_min": 6
}}

EXERCISE TO ANALYZE:
Title: {title}
Description: {description}
Type: {type}
Intensity: {intensity}

Output reasoning briefly, then JSON:
"""

# -----------------------------
# 5. Database Functions (Anti-Crash)
# -----------------------------
def get_conn_with_retry(max_retries=10):
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(NEON_PSYCO_CONN)
            conn.autocommit = False
            return conn
        except Exception as e:
            wait = (attempt + 1) * 10
            log.warning(f"Connection failed: {e}. Waiting {wait}s...")
            time.sleep(wait)
    sys.exit(1)

def fetch_unlabeled_batch(conn, limit):
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # We assume you ran the new SQL ALTER TABLE
        query = sql.SQL("""
            SELECT id, metadata_ 
            FROM {}
            WHERE details_labeled = FALSE
            LIMIT %s;
        """).format(sql.Identifier(TABLE_NAME))
        cur.execute(query, (limit,))
        return cur.fetchall()

def get_remaining_count(conn):
    with conn.cursor() as cur:
        query = sql.SQL("SELECT COUNT(*) FROM {} WHERE details_labeled = FALSE;").format(sql.Identifier(TABLE_NAME))
        cur.execute(query)
        return cur.fetchone()[0]

def update_workout_data(conn, row_id, data):
    with conn.cursor() as cur:
        query = sql.SQL("""
            UPDATE {}
            SET 
                mechanic = %s,
                force_type = %s,
                tier = %s,
                duration_min = %s,
                cal_per_min = %s,
                details_labeled = TRUE
            WHERE id = %s;
        """).format(sql.Identifier(TABLE_NAME))
        cur.execute(query, (
            data.get('mechanic', 'isolation'),
            data.get('force_type', 'static'),
            data.get('tier', 'secondary'),
            data.get('duration_min', 10),
            data.get('cal_per_min', 5),
            row_id
        ))

def mark_as_failed(conn, row_id):
    with conn.cursor() as cur:
        query = sql.SQL("UPDATE {} SET details_labeled = TRUE WHERE id = %s;").format(sql.Identifier(TABLE_NAME))
        cur.execute(query, (row_id,))

# -----------------------------
# 6. Parsing & Sanity
# -----------------------------
def parse_json_response(response_text):
    try:
        blocks = re.findall(r"\{[\s\S]*?\}", response_text)
        if not blocks: return None
        json_str = re.sub(r",\s*}", "}", blocks[-1])
        data = json.loads(json_str)
        return {
            "mechanic": str(data.get("mechanic", "isolation")).lower(),
            "force_type": str(data.get("force_type", "static")).lower(),
            "tier": str(data.get("tier", "secondary")).lower(),
            "duration_min": int(data.get("duration_min", 10)),
            "cal_per_min": int(data.get("cal_per_min", 5))
        }
    except:
        return None

def is_sane(data):
    # Basic validation
    if not data: return False
    if data['duration_min'] > 120: return False # Unlikely for one exercise
    if data['mechanic'] not in ['compound', 'isolation', 'isometric']: return False
    return True

# -----------------------------
# 7. Main Loop
# -----------------------------
def main():
    log.info("--- Starting Task 2: Science-Based Workout Extraction ---")
    conn = get_conn_with_retry()
    
    try:
        total = get_remaining_count(conn)
        log.info(f"Found {total} workouts to process.")
    except Exception:
        log.error("New columns missing! Run the SQL ALTER TABLE command first.")
        conn.close()
        return

    pbar = tqdm(total=total, desc="Classifying Exercises")
    batch_counter = 0

    while True:
        if batch_counter > 0 and batch_counter % RESET_CONNECTION_EVERY_N_BATCHES == 0:
            conn.commit()
            conn.close()
            conn = get_conn_with_retry()

        try:
            batch = fetch_unlabeled_batch(conn, BATCH_SIZE)
            if not batch: break

            for row in batch:
                r_id = row['id']
                meta = row['metadata_']
                
                title = meta.get('title', '')
                desc = meta.get('description', '')[:300]
                
                prompt = EXTRACTION_PROMPT.format(
                    title=title, description=desc, 
                    type=meta.get('type',''), intensity=meta.get('intensity','')
                )

                valid_data = None
                for _ in range(2):
                    try:
                        resp = Settings.llm.complete(prompt)
                        valid_data = parse_json_response(str(resp))
                        if is_sane(valid_data):
                            break
                    except:
                        time.sleep(1)
                
                if valid_data:
                    update_workout_data(conn, r_id, valid_data)
                else:
                    mark_as_failed(conn, r_id)

            conn.commit()
            pbar.update(len(batch))
            batch_counter += 1

        except Exception as e:
            log.error(f"Error: {e}")
            try:
                conn = get_conn_with_retry()
            except:
                break

    pbar.close()
    conn.close()

if __name__ == "__main__":
    main()