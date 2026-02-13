import pandas as pd
import psycopg2
import psycopg2.extras
import uuid
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import math

# --- Configuration ---
NEON_DB_URL = "postgresql://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"
# --- v1.1 FIX: Corrected file path ---
CSV_PATH = "archive/megaGymDataset.csv"
# -----------------------------------
TABLE_NAME = "data_workout_data"
# --- Embedding Model ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


#
def ingest_workouts():
    print(f"--- Starting Ingestion for {TABLE_NAME} ---")
    
    # 1. Load CSV
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} rows from {CSV_PATH}")
    except Exception as e:
        print(f"🔴 Error reading CSV: {e}")
        return

    # 2. Load Embedding Model (Local Brain)
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    # 3. Connect to DB
    try:
        conn = psycopg2.connect(NEON_DB_URL)
        cur = conn.cursor()
        print("✅ Connected to Neon DB.")
    except Exception as e:
        print(f"🔴 DB Connection Error: {e}")
        return

    # 4. Process Rows
    print("Processing and inserting rows...")
    inserted_count = 0
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # --- Smart Text Construction ---
            title = str(row['Title']).strip()
            desc = row['Desc']
            
            # Handle missing description (The "NaN" fix)
            if pd.isna(desc) or desc == "":
                # Fallback: Create a description from metadata
                clean_desc = f"A {row['Level']} level {row['Type']} exercise targeting the {row['BodyPart']}."
            else:
                clean_desc = str(desc).strip()

            text_content = f"{title}: {clean_desc}"
            
            # --- Metadata Construction ---
            meta = {
                "title": title,
                "description": clean_desc,
                "intensity": str(row['Level']).lower(),      
                "type": str(row['Type']).lower(),            
                "target_muscle": str(row['BodyPart']).lower(), 
                "equipment": str(row.get('Equipment', 'unknown')).lower()
            }
            
            # Generate Embedding
            embedding = model.encode(text_content).tolist()
            
            # Create UUID
            record_id = str(uuid.uuid4())
            
            # SQL Insert
            insert_sql = f"""
            INSERT INTO {TABLE_NAME} (id, text, metadata_, embedding, node_id, ref_doc_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
            """
            
            cur.execute(insert_sql, (
                record_id,
                text_content,
                json.dumps(meta),
                embedding,
                record_id,
                record_id
            ))
            inserted_count += 1
            
        except Exception as row_err:
            continue

    conn.commit()
    print(f"✅ Successfully ingested {inserted_count} workouts into {TABLE_NAME}.")
    cur.close()
    conn.close()

if __name__ == "__main__":
    ingest_workouts()