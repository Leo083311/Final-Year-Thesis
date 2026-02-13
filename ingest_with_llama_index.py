#!/usr/bin/env python3
"""
ingest_with_llama_index.py

- Creates LlamaIndex-compatible table schema in Neon (pgvector)
  Columns: id, text, embedding, metadata_, node_id, ref_doc_id
- Streams/ingests nutrition CSV (optional)
- Loads recipe CSV, computes embeddings in parallel, and bulk-inserts
- Resume support (via 'doc_hash' in metadata_)
- Creates HNSW index concurrently (if available)
Requirements:
  pip install pandas sqlalchemy psycopg2-binary sentence-transformers tqdm
"""
# Standard libraries
import os
import sys
import time
import json
import hashlib
import logging
import math
# parallelize the embedding generation
import multiprocessing
import uuid  # Required for LlamaIndex schema
import psycopg2.extensions  # Required for autocommit fix
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict

import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -----------------------------
# Exact configuration (from prompt)
# -----------------------------
NEON_PSYCO_CONN = "postgresql://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"
NEON_SQLALCHEMY = "postgresql+psycopg2://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"

RECIPES_CSV = "archive/Food Ingredients and Recipe Dataset with Image Name Mapping.csv"
NUTRITION_CSV = "archive/openfoodfacts.csv"

RECIPE_TABLE = "data_recipes_data"      # LlamaIndex-compatible table name
NUTRITION_TABLE = "data_nutrition_data"
# We set the Vector Dimension to 384.
# This MUST match the AI model we use (all-MiniLM-L6-v2 output size).
# If this number is wrong, the database will reject the data.
VECTOR_DIM = 384

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Tunable parameters
BULK_INSERT_BATCH = 256
EMBED_BATCH_SIZE = 128
# This calculate how many CPU cores my computer has.
# If I have 8 cores, my laptop use 7 cores. This leaves 1 core free so I
# don't freeze while processing.
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

# Controls
RUN_NUTRITION_INGESTION = True
FORCE_RECREATE_TABLE = False  # set True to drop/create table and ignore resume

# Safety
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("ingestion.log", mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ingest")

# -----------------------------
# Helpers
# -----------------------------
# DB Connection
def get_conn():
    return psycopg2.connect(NEON_PSYCO_CONN, cursor_factory=psycopg2.extras.DictCursor)

# Checksum for resume
def checksum_doc(title: str, ingredients: str, instructions: str) -> str:
    # I create a unique "fingerprint" (MD5 Hash) for every recipe.
    # If the text is the same, this hash is identical every time.
    h = hashlib.md5()
    combined = (str(title or "") + "||" + str(ingredients or "") + "||" + str(instructions or "")).encode("utf-8")
    h.update(combined)
    return h.hexdigest()

# Ensure pgvector extension is enabled
def ensure_pgvector_extension(conn):
    with conn.cursor() as cur:
        # I created a table tailored for the LlamaIndex PGVectorStore.
        # 'embedding': I defined this column specifically for 384-dimensional vectors.
        # 'metadata_': I used JSONB here to handle unstructured data flexibly.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    log.info("Ensured pgvector extension exists.")

def create_llamaindex_table_if_missing(conn):
    """
    Creates the exact schema LlamaIndex PGVectorStore expects.
    """
    with conn.cursor() as cur:
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {RECIPE_TABLE} (
            id UUID PRIMARY KEY,
            embedding VECTOR({VECTOR_DIM}),
            text TEXT,
            metadata_ JSONB,
            node_id VARCHAR(256),
            ref_doc_id VARCHAR(256)
        );
        """)
        # Create GIN index on metadata_ for fast filtering (per mandate)
        cur.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_metadata_gin
        ON {RECIPE_TABLE} USING GIN(metadata_);
        """)
    conn.commit()
    log.info(f"Ensured LlamaIndex-compatible table '{RECIPE_TABLE}' exists.")

def drop_recipe_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {RECIPE_TABLE} CASCADE;")
    conn.commit()
    log.info(f"Dropped table {RECIPE_TABLE}.")

def get_existing_doc_hashes(conn) -> set:
    """
    Queries the metadata_ JSONB blob for our custom doc_hash.
    """
    with conn.cursor() as cur:
        # This query checks if the 'doc_hash' key exists in the JSONB
        cur.execute(f"SELECT metadata_ ->> 'doc_hash' FROM {RECIPE_TABLE} WHERE metadata_ ? 'doc_hash';")
        rows = cur.fetchall()
    existing = {r[0] for r in rows} if rows else set()
    log.info(f"Found {len(existing)} existing doc_hashes for resume.")
    return existing

# -----------------------------
# Embedding worker
# -----------------------------
def _worker_encode_texts(texts: List[str]) -> List[List[float]]:
    # Worker loads its own model (safer with multiprocessing)
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embs = model.encode(texts, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
    return [list(map(float, v)) for v in embs]

# -----------------------------
# Nutrition ingestion (streamed)
# -----------------------------
# My solution for handling the massive "OpenFoodFacts" dataset without crashing the system
def ingest_nutrition():
    # So that I have the option to turn it on or off
    if not RUN_NUTRITION_INGESTION:
        log.info("Skipping nutrition ingestion.")
        return

    log.info("Starting nutrition ingestion (streamed)...")
    try:
        # By filtering out hundreds of unused columns before pandas even reads the file, I drastically reduced the memory footprint and processing time
        use_cols = [
            'code', 'product_name', 'nutriscore_grade', 'energy-kcal_100g',
            'fat_100g', 'carbohydrates_100g', 'proteins_100g', 'salt_100g', 'sodium_100g'
        ]
        engine = create_engine(NEON_SQLALCHEMY)
        # I use 'chunksize' in read_csv. This turns the function into an Iterator.
        # Instead of returning one giant DataFrame, it yields smaller DataFrames of 20,000 rows.
        chunk_size = 20000
        first = True
        for chunk in pd.read_csv(NUTRITION_CSV, low_memory=False, on_bad_lines='skip',
                                 usecols=lambda c: c in use_cols, chunksize=chunk_size, encoding='utf-8'):
            chunk.to_sql(NUTRITION_TABLE, con=engine, index=False,
                         if_exists='replace' if first else 'append', method="multi")
            first = False
            log.info(f"Wrote nutrition chunk of {len(chunk)} rows.")
        log.info(f"✅ Nutrition ingestion completed to table '{NUTRITION_TABLE}'.")
    # A failure here should log an error but not crash the entire pipeline
    except Exception:
        log.exception("Nutrition ingestion failed but continuing...")

# -----------------------------
# Recipe ingestion
# -----------------------------
def ingest_recipes():
    log.info("Starting recipe ingestion with resume + parallel embeddings...")

    if not os.path.exists(RECIPES_CSV):
        log.error(f"Recipe CSV not found at path: {RECIPES_CSV}")
        sys.exit(1)

    # To prevent the script from crashing on corrupted CSV lines, 'on_bad_lines' is set to 'skip'.
    # NaNs are filled with empty strings to avoid TypeErrors during string concatenation.
    df = pd.read_csv(RECIPES_CSV, low_memory=False, encoding='utf-8', on_bad_lines='skip')
    df = df.fillna("")
    log.info(f"Loaded {len(df)} rows from recipes CSV.")

    # To allow flexibility in input formats, a helper function auto-detects column names
    def choose_col(options):
        for o in options:
            if o in df.columns:
                return o
        return None

    title_col = choose_col(["Title", "title", "recipe_name", "name"])
    ingredients_col = choose_col(["Cleaned_Ingredients", "Cleaned-Ingredients", "Cleaned Ingredients", "Ingredients", "ingredients"])
    instructions_col = choose_col(["Instructions", "instructions", "Method", "method"])
    image_col = choose_col(["Image_Name", "Image Name", "image_name", "image"])
    orig_ingredients_col = choose_col(["Ingredients", "ingredients_str"]) # For preference filter

    if not title_col or not ingredients_col:
        log.warning("Missing expected title or ingredients columns. Proceeding.")

    # Build candidate rows for LlamaIndex schema
    candidates = []
    for _, row in df.iterrows():
        title = row.get(title_col, "") if title_col else ""
        cleaned_ingredients = row.get(ingredients_col, "") if ingredients_col else ""
        instructions = row.get(instructions_col, "") if instructions_col else ""
        
        # Generate a UUID for the vector database ID.
        doc_hash = checksum_doc(title, cleaned_ingredients, instructions)
        node_uuid = uuid.uuid4()
        document_text = f"Title: {title}\nIngredients: {cleaned_ingredients}\nInstructions: {instructions}"
        
        # To structure the metadata specifically for the database JSONB column.
        # Empty lists are initialized for 'health_labels' and 'goal_labels' to prepare
        # the structure for future enrichment steps (Component B).
        metadata = {
            "title": title,
            "ingredients_str": row.get(orig_ingredients_col, ""), # For Filter 3
            "cleaned_ingredients_str": cleaned_ingredients,
            "instructions": instructions,
            "image_name": row.get(image_col, "") if image_col else "",
            "doc_hash": doc_hash, # Our custom field for resuming
            "health_labels": [], # Placeholder for Component B
            "goal_labels": []    # Placeholder for Component B
        }
        # (node_uuid, document_text, metadata, doc_hash)
        candidates.append((node_uuid, document_text, metadata, doc_hash))

    log.info(f"Prepared {len(candidates)} candidate documents (LlamaIndex schema).")

    conn = get_conn()
    try:
        ensure_pgvector_extension(conn)
        if FORCE_RECREATE_TABLE:
            drop_recipe_table(conn)
        create_llamaindex_table_if_missing(conn) # Creates the new schema
        # To identify records that have already been processed.
        existing_hashes = get_existing_doc_hashes(conn)
    except Exception:
        log.exception("Fatal DB setup error.")
        conn.close()
        sys.exit(1)

    # To filter out candidates that already exist in the database.
    new_items = [item for item in candidates if item[3] not in existing_hashes] # item[3] is doc_hash
    log.info(f"After resume: {len(new_items)} documents to ingest (skipped {len(candidates) - len(new_items)}).")

    # To prevent unnecessary processing if no new data is found.
    if not new_items:
        conn.close()
        log.info("No new documents to ingest. Exiting recipe ingestion.")
        return

    texts = [it[1] for it in new_items] # it[1] is document_text
    total = len(texts)

    # Worker chunking
    workers = min(NUM_WORKERS, total)
    chunk_size = math.ceil(total / workers)
    # To distribute the workload evenly across available CPU cores.
    chunks = [texts[i:i + chunk_size] for i in range(0, total, chunk_size)]

    log.info(f"Computing embeddings in parallel with {workers} worker(s) (model: {EMBED_MODEL_NAME})...")
    embeddings = []
    start_t = time.time()
    # To execute the embedding task in parallel processes.
    with ProcessPoolExecutor(max_workers=workers) as exe:
        # To map each chunk of text to a worker process for independent conversion.
        futures = {exe.submit(_worker_encode_texts, chunk): idx for idx, chunk in enumerate(chunks)}
        chunk_results = [None] * len(chunks)
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Embedding chunks"):
            idx = futures[fut]
            try:
                result = fut.result()
                chunk_results[idx] = result
                log.info(f"Worker finished chunk {idx} (size {len(result)})")
            except Exception:
                log.exception("Worker failed; falling back to main-process encoding for this chunk.")
                # fallback: compute in main process
                chunk = chunks[idx]
                model = SentenceTransformer(EMBED_MODEL_NAME)
                result = model.encode(chunk, batch_size=EMBED_BATCH_SIZE, show_progress_bar=False, convert_to_numpy=True)
                chunk_results[idx] = [list(map(float, v)) for v in result]

    # flatten, preserving order
    for ch in chunk_results:
        embeddings.extend(ch)

    elapsed = time.time() - start_t
    log.info(f"Embedding computation completed in {elapsed:.1f}s. Total embeddings: {len(embeddings)}")

    if len(embeddings) != len(new_items):
        log.error("Embedding count mismatch; aborting.")
        conn.close()
        sys.exit(1)

    # Bulk insert batches for LlamaIndex schema
    insert_sql = f"""
    INSERT INTO {RECIPE_TABLE} (id, text, metadata_, embedding, node_id, ref_doc_id)
    VALUES %s
    ON CONFLICT (id) DO NOTHING;
    """
    cur = conn.cursor()
    inserted = 0
    try:
        for i in tqdm(range(0, len(new_items), BULK_INSERT_BATCH), desc="DB insert batches"):
            batch_items = new_items[i:i + BULK_INSERT_BATCH]
            batch_embs = embeddings[i:i + BULK_INSERT_BATCH]
            values = []
            # item = (node_uuid, document_text, metadata, doc_hash)
            for (node_uuid, doc_text, metadata, _), emb in zip(batch_items, batch_embs):
                emb_str = "[" + ",".join(map(lambda x: repr(float(x)), emb)) + "]"
                # (id, text, metadata_, embedding, node_id, ref_doc_id)
                values.append((
                    node_uuid, 
                    doc_text, 
                    psycopg2.extras.Json(metadata), # Note: metadata_ column
                    emb_str, 
                    str(node_uuid), # Populate node_id
                    str(node_uuid)  # Populate ref_doc_id
                ))
            template = "(%s, %s, %s, %s::vector, %s, %s)"
            # To insert data in efficient batches (e.g., 256 rows) rather than one by one.
            # To explicitly cast the embedding string to the PostgreSQL 'vector' type.
            psycopg2.extras.execute_values(cur, insert_sql, values, template=template)
            conn.commit()
            inserted += len(values)
            log.info(f"Inserted batch of {len(values)} rows (total inserted so far: {inserted})")
    except Exception:
        log.exception("Bulk insert failed; rolling back and aborting.")
        conn.rollback()
        cur.close()
        conn.close()
        sys.exit(1)

    cur.close()

    # Verification
    try:
        with conn.cursor() as vcur:
            vcur.execute(f"SELECT COUNT(*) FROM {RECIPE_TABLE};")
            total_count = vcur.fetchone()[0]
        log.info(f"Verification: table {RECIPE_TABLE} contains {total_count} rows (after insertion).")
    except Exception:
        log.exception("Verification query failed.")
    finally:
        conn.close()

    log.info("Recipe ingestion completed successfully.")

# -----------------------------
# Create vector index (HNSW if supported)
# -----------------------------
def create_vector_index():
    log.info("Creating HNSW index concurrently (if pgvector supports HNSW).")
    conn = get_conn()
    
    # --- THIS IS THE FIX ---
    # We must set isolation_level to AUTOCOMMIT to run
    # CREATE INDEX CONCURRENTLY, which cannot be in a transaction block.
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    # -----------------------

    cur = conn.cursor()
    try:
        # Set a reasonable timeout for index creation
        cur.execute("SET statement_timeout = '300s';") # 5 minutes

        # --- THIS IS THE SECOND FIX ---
        # Use a simple, direct SQL command.
        # This is idempotent (IF NOT EXISTS) and works with CONCURRENTLY.
        create_index_sql = f"""
        CREATE INDEX CONCURRENTLY IF NOT EXISTS {RECIPE_TABLE}_embedding_idx
        ON {RECIPE_TABLE}
        USING HNSW (embedding vector_l2_ops)
        WITH (m = 16, ef_construction = 64);
        """
        
        cur.execute(create_index_sql)
        log.info("✅ Index creation command executed successfully.")
    except Exception:
        log.exception("Index creation failed. Your pgvector version may not support HNSW.")
    finally:
        cur.close()
        conn.close()

# -----------------------------
# Main
# -----------------------------
def main():
    # Shows logs so i can tract the progress
    start = time.time()
    log.info("=== Starting ingestion pipeline ===")

    # List tables
    try:
        conn = get_conn()
        # To confirm connectivity and log the current state of the database.
        # This helps debug issues immediately (e.g., if the 'recipes' table is missing).
        with conn.cursor() as c:
            c.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            names = [r[0] for r in c.fetchall()]
        conn.close()
        log.info("Existing public tables: " + ", ".join(names))
    except Exception:
        log.exception("Could not list public tables.")

    # This is independent of recipes, but required for the full system.
    ingest_nutrition()

    # This includes text cleaning, vector embedding, and database insertion.
    ingest_recipes()

    # Indexing is faster and more efficient once the data is already in place.
    create_vector_index()

    elapsed = time.time() - start
    log.info(f"=== Ingestion finished in {elapsed:.1f}s ===")

if __name__ == "__main__":
    main()