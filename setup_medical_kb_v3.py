import psycopg2
from psycopg2 import sql
import sys

NEON_DB_URL = "postgresql://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"
TABLE_NAME = "data_medical_knowledge"

def setup_medical_kb():
    print(f"\n--- Creating / Updating Medical KB Table '{TABLE_NAME}' (v3 merged) ---\n")
    conn = None

    try:
        conn = psycopg2.connect(NEON_DB_URL)
        cur = conn.cursor()

        # ----------------------------------------------------------
        # 1. CREATE TABLE (Merged Schema)
        # ----------------------------------------------------------
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {} (
                id SERIAL PRIMARY KEY,
                condition_name VARCHAR(255) UNIQUE NOT NULL,

                severity VARCHAR(20) DEFAULT 'medium' NOT NULL,

                apply_recipe_health TEXT[] DEFAULT '{{}}'::text[] NOT NULL,
                avoid_recipe_health TEXT[] DEFAULT '{{}}'::text[] NOT NULL,
                avoid_recipe_goals  TEXT[] DEFAULT '{{}}'::text[] NOT NULL,
                avoid_workout_intensity TEXT[] DEFAULT '{{}}'::text[] NOT NULL,

                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
        """).format(sql.Identifier(TABLE_NAME)))

        print("✅ Schema verified.\n")

        # ----------------------------------------------------------
        # 2. MERGED + CLEANED MEDICAL KNOWLEDGE BASE (25 CONDITIONS)
        # ----------------------------------------------------------
        # Each tuple MUST have 6 items to match the 6 placeholders
        knowledge = [
            # (condition_name, severity, apply_health, avoid_health, avoid_goals, avoid_intensity)
            
            # --- METABOLIC ---
            ("Type 2 Diabetes", "HIGH", ["diabetic_friendly"], [], ["keto_friendly"], []),
            ("Type 1 Diabetes", "HIGH", ["diabetic_friendly"], [], ["keto_friendly"], []),
            ("Prediabetes", "MEDIUM", ["diabetic_friendly", "low_glycemic"], [], [], []),
            ("Hypothyroidism", "MEDIUM", [], ["gluten_free"], [], []),
            ("Polycystic Ovary Syndrome (PCOS)", "MEDIUM", ["low_glycemic"], [], [], []),
            ("Obesity (BMI 30+)", "MEDIUM", ["low_fat"], [], [], []),

            # --- CARDIOVASCULAR ---
            ("Hypertension", "HIGH", ["low_sodium", "heart_healthy"], [], [], ["high"]),
            ("High Cholesterol", "MEDIUM", ["low_fat", "heart_healthy"], [], [], []),
            ("Coronary Artery Disease", "HIGH", ["low_fat", "heart_healthy", "low_sodium"], [], [], ["high"]),
            ("Heart Failure", "HIGH", ["low_sodium", "heart_healthy"], [], [], ["high", "medium"]),
            ("Atrial Fibrillation", "HIGH", ["heart_healthy"], [], [], ["high", "medium"]),

            # --- GASTROINTESTINAL ---
            ("Celiac Disease", "HIGH", ["gluten_free"], [], [], []),
            ("GERD / Acid Reflux", "MEDIUM", [], ["spicy", "acidic", "high_fat"], [], []),
            ("Irritable Bowel Syndrome (IBS)", "MEDIUM", [], ["high_fodmap"], [], []),
            ("Lactose Intolerance", "LOW", ["dairy_free"], [], [], []),
            ("Non-Alcoholic Fatty Liver Disease (NAFLD)", "MEDIUM", ["low_fat", "diabetic_friendly"], ["high_fat"], [], []),

            # --- RENAL ---
            ("Chronic Kidney Disease (CKD)", "HIGH", ["low_sodium"], ["high_potassium", "high_phosphorus"], ["high_protein"], ["high"]),

            # --- MUSCULOSKELETAL ---
            ("Osteoarthritis", "MEDIUM", [], [], [], ["high"]),
            ("Lower Back Injury", "MEDIUM", [], [], [], ["high", "medium"]),
            ("Knee Injury", "MEDIUM", [], [], [], ["high"]),

            # --- RESPIRATORY ---
            ("Asthma", "MEDIUM", [], [], [], ["high"]),
            ("COPD", "HIGH", [], [], [], ["high", "medium"]),

            # --- AUTOIMMUNE ---
            ("Rheumatoid Arthritis", "MEDIUM", ["anti_inflammatory"], [], [], ["high"]),
            ("Lupus", "MEDIUM", ["anti_inflammatory"], [], [], ["high"]),
            ("Hashimoto's Thyroiditis", "MEDIUM", [], ["gluten_free"], [], [])
        ]

        # ----------------------------------------------------------
        # 3. INSERT WITH UPSERT (MERGED ADVANTAGE)
        # ----------------------------------------------------------
        insert_query = sql.SQL("""
            INSERT INTO {} (
                condition_name, severity,
                apply_recipe_health,
                avoid_recipe_health,
                avoid_recipe_goals,
                avoid_workout_intensity
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (condition_name) DO UPDATE SET
                severity = EXCLUDED.severity,
                apply_recipe_health = EXCLUDED.apply_recipe_health,
                avoid_recipe_health = EXCLUDED.avoid_recipe_health,
                avoid_recipe_goals = EXCLUDED.avoid_recipe_goals,
                avoid_workout_intensity = EXCLUDED.avoid_workout_intensity,
                updated_at = NOW();
        """).format(sql.Identifier(TABLE_NAME))

        cur.executemany(insert_query, knowledge)
        conn.commit()

        cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(TABLE_NAME)))
        total = cur.fetchone()[0]

        print(f"✅ KB populated successfully. Total conditions: {total}\n")

    except Exception as e:
        print(f"❌ ERROR: {e}\n")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    setup_medical_kb()