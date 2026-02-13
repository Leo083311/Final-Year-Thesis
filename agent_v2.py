import os
import json
import psycopg2
import psycopg2.extras 
from psycopg2 import sql
from dotenv import load_dotenv
from typing import List, Dict, Optional, TypedDict

# --- Core Libraries ---
import google.generativeai as genai
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

# --- V1.0 RAG Dependencies ---
from sentence_transformers import SentenceTransformer
from llama_index.core.schema import NodeWithScore, TextNode 

# -------------------------------------------------
# 1. Configuration & Setup
# -------------------------------------------------

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
NEON_DB_URL = "postgresql://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"

if not GOOGLE_API_KEY:
    raise SystemExit("Error: GOOGLE_API_KEY not found in .env file")

print("--- AI Studio Key Loaded ---")
genai.configure(api_key=GOOGLE_API_KEY)
PATIENT_FILE = "medical_report.json"

# --- V1.0 RAG Configuration ---
PRE_FETCH_K = 50 
RECIPE_TABLE = "data_recipes_data"
WORKOUT_TABLE = "data_workout_data" 
HEALTH_LABEL_KEY = "health_labels"
GOAL_LABEL_KEY = "goal_labels"    
WORKOUT_INTENSITY_KEY = "intensity" 

print("--- Loading Embedding Model (all-MiniLM-L6-v2)... ---")
try:
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Embedding Model Loaded.")
except Exception as e:
    raise SystemExit(f"🔴 CRITICAL ERROR: Failed to load SentenceTransformer: {e}")

# --- Initialize Gemini Model (for final node) ---
# --- Initialize Gemini Model (for final node) ---
try:
    gemini_model = genai.GenerativeModel('gemini-2.5-flash') # <-- THIS IS THE FIX
    print("✅ Gemini Model Initialized.")
except Exception as e:
    raise SystemExit(f"🔴 CRITICAL ERROR: Failed to initialize Gemini: {e}")

# -------------------------------------------------
# 2. Define the Agent's "Memory" (The State)
# -------------------------------------------------

class AgentState(TypedDict):
    user_query: str                  
    patient_profile: Dict            
    contraindications: Dict 
    bmr_tdee: Dict 
    recipe_candidates: List[Dict] 
    workout_candidates: List[Dict] 
    final_recommendation: str        

# -------------------------------------------------
# 3. V1.0 RAG Functions (Tools)
# -------------------------------------------------

def fetch_recipe_nodes(query_str: str, top_k: int) -> List[NodeWithScore]:
    """ Connects via psycopg2 and runs a raw SQL vector query on the RECIPE table. """
    unfiltered_nodes = []
    conn, cur = None, None
    try:
        query_embedding = embed_model.encode(query_str).tolist()
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        conn = psycopg2.connect(NEON_DB_URL)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        query = sql.SQL("""
        SELECT id, text, metadata_, (embedding <-> %s) AS distance 
        FROM {} ORDER BY distance LIMIT %s;
        """).format(sql.Identifier(RECIPE_TABLE))
        
        cur.execute(query, (embedding_str, top_k))
        results = cur.fetchall()

        for row in results:
            node = TextNode(id_=str(row['id']), text=row['text'], metadata=row['metadata_'])
            unfiltered_nodes.append(NodeWithScore(node=node, score=(1.0 - row['distance'])))
    except Exception as e:
        print(f"🔴 ERROR during raw SQL query (Recipes): {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()
    return unfiltered_nodes

def fetch_workout_nodes(query_str: str, top_k: int) -> List[NodeWithScore]:
    """ Connects via psycopg2 and runs a raw SQL vector query on the WORKOUT table. """
    unfiltered_nodes = []
    conn, cur = None, None
    try:
        query_embedding = embed_model.encode(query_str).tolist()
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        conn = psycopg2.connect(NEON_DB_URL)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        query = sql.SQL("""
        SELECT id, text, metadata_, (embedding <-> %s) AS distance 
        FROM {} ORDER BY distance LIMIT %s;
        """).format(sql.Identifier(WORKOUT_TABLE))
        
        cur.execute(query, (embedding_str, top_k))
        results = cur.fetchall()

        for row in results:
            node = TextNode(id_=str(row['id']), text=row['text'], metadata=row['metadata_'])
            unfiltered_nodes.append(NodeWithScore(node=node, score=(1.0 - row['distance'])))
    except Exception as e:
        print(f"🔴 ERROR during raw SQL query (Workouts): {e}")
    finally:
        if cur: cur.close()
        if conn: conn.close()
    return unfiltered_nodes


def manual_filter_recipes(nodes: List[NodeWithScore], health_labels: List[str], goal_labels: List[str], avoid_health_labels: List[str]) -> List[NodeWithScore]:
    """ 
    Manually filters RECIPE nodes.
    APPLIES health_labels and goal_labels.
    AVOIDS avoid_health_labels.
    """
    filtered_nodes = []
    print(f"--- Applying RECIPE filter: ---")
    print(f"    MUST HAVE (Health): {health_labels}")
    print(f"    MUST HAVE (Goals):  {goal_labels}")
    print(f"    MUST NOT HAVE:      {avoid_health_labels}")
    
    for node_with_score in nodes:
        meta = node_with_score.node.metadata
        node_health = meta.get(HEALTH_LABEL_KEY, [])
        node_goals = meta.get(GOAL_LABEL_KEY, [])
        
        # 1. Check for required labels
        health_ok = all(h in node_health for h in health_labels)
        goal_ok = all(g in node_goals for g in goal_labels)
        
        # 2. Check for forbidden labels
        # any() returns True if any forbidden label is found
        avoid_ok = not any(h in node_health for h in avoid_health_labels)
        
        if health_ok and goal_ok and avoid_ok:
            filtered_nodes.append(node_with_score)
            
    print(f"--- Recipe Filter Result: {len(filtered_nodes)} nodes passed. ---")
    return filtered_nodes

def manual_filter_workouts(nodes: List[NodeWithScore], avoid_intensity: List[str]) -> List[NodeWithScore]:
    """ Manually filters WORKOUT nodes based on intensity. """
    filtered_nodes = []
    print(f"--- Applying WORKOUT filter: Avoid Intensity={avoid_intensity} ---")
    for node_with_score in nodes:
        meta = node_with_score.node.metadata
        intensity = meta.get(WORKOUT_INTENSITY_KEY, "unknown")
        
        if intensity not in avoid_intensity:
            filtered_nodes.append(node_with_score)
            
    print(f"--- Workout Filter Result: {len(filtered_nodes)} nodes passed. ---")
    return filtered_nodes

# -------------------------------------------------
# 4. Define the Agent's "Nodes" (The Functions)
# -------------------------------------------------

def load_patient_profile(state: AgentState):
    """ Node 1: Loads the simulated medical_report.json """
    print("--- 1. Node: load_patient_profile ---")
    try:
        with open(PATIENT_FILE, 'r') as f:
            profile = json.load(f)
        print(f"✅ Loaded patient profile for: {profile.get('name')}")
        return {"patient_profile": profile}
    except Exception as e:
        print(f"🔴 CRITICAL ERROR: {e}")
        return {"patient_profile": {}}

def get_contraindications(state: AgentState):
    """ Node 2: Queries data_medical_knowledge. (Complete) """
    print("--- 2. Node: get_contraindications (Live) ---")
    conditions = state.get('patient_profile', {}).get('conditions', [])
    if not conditions: return {"contraindications": {}}

    merged_rules = {
        "apply_recipe_health": set(), "avoid_recipe_health": set(),
        "avoid_recipe_goals": set(), "avoid_workout_intensity": set()
    }
    conn, cur = None, None
    try:
        conn = psycopg2.connect(NEON_DB_URL)
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        query = "SELECT * FROM data_medical_knowledge WHERE condition_name = ANY(%s);"
        cur.execute(query, (conditions,))
        rows = cur.fetchall()
        print(f"✅ Found {len(rows)} matching rules in KB for {conditions}")
        for row in rows:
            merged_rules["apply_recipe_health"].update(row['apply_recipe_health'])
            merged_rules["avoid_recipe_health"].update(row['avoid_recipe_health'])
            merged_rules["avoid_recipe_goals"].update(row['avoid_recipe_goals'])
            merged_rules["avoid_workout_intensity"].update(row['avoid_workout_intensity'])
        final_rules = {k: list(v) for k, v in merged_rules.items()}
        print(f"✅ Merged rules: {final_rules}")
        return {"contraindications": final_rules}
    except Exception as e:
        print(f"🔴 CRITICAL ERROR: DB query failed: {e}")
        return {"contraindications": {}}
    finally:
        if cur: cur.close()
        if conn: conn.close()

def calculate_bmr(state: AgentState):
    """ Node 3: Calculates BMR and TDEE. (Complete) """
    print("--- 3. Node: calculate_bmr (Live) ---")
    try:
        profile = state['patient_profile']
        age, sex, weight_kg, height_cm = profile['age'], profile['sex'].lower(), profile['weight_kg'], profile['height_cm']
        
        if sex == 'male': bmr = 88.362 + (13.397 * weight_kg) + (4.799 * height_cm) - (5.677 * age)
        else: bmr = 447.593 + (9.247 * weight_kg) + (3.098 * height_cm) - (4.330 * age)
        bmr = round(bmr)
        
        multipliers = {"sedentary": 1.2, "lightly_active": 1.375, "moderately_active": 1.55}
        multiplier = multipliers.get(profile.get("activity_level", "lightly_active"), 1.375)
        tdee = round(bmr * multiplier)
        
        calorie_target = tdee
        if "Weight Loss" in profile.get("dietary_goals", []):
            calorie_target -= 500
        
        metrics = {"bmr": bmr, "tdee_maintenance": tdee, "calorie_target": calorie_target}
        print(f"✅ Calculated metrics: {metrics}")
        return {"bmr_tdee": metrics}
    except Exception as e:
        print(f"🔴 CRITICAL ERROR: BMR calculation failed: {e}")
        return {"bmM_tdee": {}}

# --- THIS NODE IS NOW FIXED ---
def call_recipe_tool(state: AgentState):
    """
    Node 4: Queries data_recipes_data with CORRECT safety logic.
    """
    print("--- 4. Node: call_recipe_tool (Live, v2.5) ---")
    
    profile = state.get('patient_profile', {})
    rules = state.get('contraindications', {})
    
    # 1. Formulate search query (unchanged)
    query_parts = profile.get('dietary_goals', []) + profile.get('conditions', [])
    search_query = " ".join(query_parts)
    print(f"ℹ️ Smart query: '{search_query}'")
    
    # 2. Get all labels and rules
    apply_health = set(rules.get('apply_recipe_health', []))
    avoid_health = set(rules.get('avoid_recipe_health', []))
    avoid_goals = set(rules.get('avoid_recipe_goals', []))
    patient_goals = set(profile.get('dietary_goals', []))

    # 3. CONFLICT RESOLUTION (The Correct Logic)
    
    # 3a. Map patient goals to their corresponding labels
    goal_to_label_map = {
        "Muscle Gain": "high_protein",
        "Weight Loss": "low_carb", # Or "low_fat"
        # Add more mappings as needed
    }
    
    patient_goal_labels = set()
    for goal in patient_goals:
        if goal in goal_to_label_map:
            patient_goal_labels.add(goal_to_label_map[goal])

    # 3b. Calculate the *safe* goal labels
    # (Patient Goal Labels) - (Avoid Goal Labels)
    safe_goal_labels = patient_goal_labels - avoid_goals
    
    # 4. Define final filters
    final_apply_health = list(apply_health)
    final_apply_goals = list(safe_goal_labels)
    final_avoid_health = list(avoid_health)
    
    print(f"ℹ️ Patient Goal Labels: {patient_goal_labels}")
    print(f"ℹ️ Avoid Goal Labels:   {avoid_goals}")
    print(f"ℹ️ Safe Goal Labels:    {safe_goal_labels}")
    
    # 5. Fetch wide-net candidates
    unfiltered_nodes = fetch_recipe_nodes(search_query, PRE_FETCH_K)
    print(f"✅ Found {len(unfiltered_nodes)} raw recipe candidates.")
    
    # 6. Manually filter candidates with all rules
    safe_nodes = manual_filter_recipes(
        unfiltered_nodes, 
        final_apply_health, 
        final_apply_goals,
        final_avoid_health # Pass the avoid list
    )
    
    final_candidates = [node.node.to_dict() for node in safe_nodes]
    return {"recipe_candidates": final_candidates}

# --- THIS NODE IS NOW LIVE ---
def call_workout_tool(state: AgentState):
    """
    Node 5: Queries data_workout_data using safety rules.
    """
    print("--- 5. Node: call_workout_tool (Live, v2.5) ---")
    
    profile = state.get('patient_profile', {})
    rules = state.get('contraindications', {})
    
    # 1. Formulate search query
    query_parts = profile.get('dietary_goals', []) 
    search_query = " ".join(query_parts)
    print(f"ℹ️ Smart query: '{search_query}'")

    # 2. Get safety rules
    avoid_intensity = rules.get('avoid_workout_intensity', [])
    
    # 3. Fetch wide-net candidates
    unfiltered_nodes = fetch_workout_nodes(search_query, PRE_FETCH_K)
    print(f"✅ Found {len(unfiltered_nodes)} raw workout candidates.")

    # 4. Manually filter candidates
    safe_nodes = manual_filter_workouts(unfiltered_nodes, avoid_intensity)
    
    final_candidates = [node.node.to_dict() for node in safe_nodes]
    return {"workout_candidates": final_candidates}

# --- THIS NODE IS NOW LIVE (FINAL) ---
def rank_and_synthesize(state: AgentState):
    """
    Node 6: Calls Gemini API to rank candidates and generate final answer.
    """
    print("--- 6. Node: rank_and_synthesize (Live) ---")
    
    profile = state.get('patient_profile', {})
    rules = state.get('contraindications', {})
    metrics = state.get('bmr_tdee', {})
    recipes = state.get('recipe_candidates', [])
    workouts = state.get('workout_candidates', [])
    
    # 1. Create a detailed prompt for the Gemini Brain
    prompt = f"""
    You are a world-class AI medical advisor and nutritionist.
    Your job is to provide a safe, personalized, and empathetic recommendation
    based on the user's complex profile and a pre-filtered list of safe options.

    --- PATIENT PROFILE ---
    {json.dumps(profile, indent=2)}

    --- CALCULATED METRICS ---
    {json.dumps(metrics, indent=2)}

    --- SAFETY RULES APPLIED ---
    {json.dumps(rules, indent=2)}

    --- PRE-FILTERED SAFE RECIPES ({len(recipes)} found) ---
    {json.dumps([r['metadata']['title'] for r in recipes[:10]], indent=2)} 
    
    --- PRE-FILTERED SAFE WORKOUTS ({len(workouts)} found) ---
    {json.dumps([w['metadata']['title'] for w in workouts[:10]], indent=2)}

    --- YOUR TASK ---
    Generate a final recommendation for the user, Leonard Tye.
    
    1.  **Acknowledge the user's goals** (e.g., "Muscle Gain" and "Weight Loss").
    2.  **Explain the conflict (CRITICAL):** State clearly that because of their
        Chronic Kidney Disease (CKD), a "high_protein" diet (for Muscle Gain)
        is unsafe. You must prioritize their kidney health.
    3.  **State the Plan:** Explain that you have created a plan that supports
        "Weight Loss" (calorie target: {metrics.get('calorie_target')} kcal)
        while adhering to *all* their medical safety rules (CKD, Hypertension, GERD).
    4.  **Recommend 1-2 Recipes:** Pick the best-sounding recipes from the safe list.
    5.  **Recommend 1-2 Workouts:** Pick the best-sounding workouts from the safe list.
    6.  Keep the tone professional, empathetic, and clear.
    """
    
    try:
        print("--- Calling Gemini API for final ranking... ---")
        response = gemini_model.generate_content(prompt)
        final_text = response.text
        print("✅ Gemini ranking complete.")
    except Exception as e:
        print(f"🔴 CRITICAL ERROR: Gemini API call failed: {e}")
        final_text = "I'm sorry, I was unable to generate a final recommendation."

    return {"final_recommendation": final_text}

# -------------------------------------------------
# 5. Define the Graph (The "Brain's" Logic)
# -------------------------------------------------

print("--- Building Agent Graph (v2.5) ---")
workflow = StateGraph(AgentState)

workflow.add_node("load_patient_profile", load_patient_profile)
workflow.add_node("get_contraindications", get_contraindications) 
workflow.add_node("calculate_bmr", calculate_bmr)
workflow.add_node("call_recipe_tool", call_recipe_tool) 
workflow.add_node("call_workout_tool", call_workout_tool) 
workflow.add_node("rank_and_synthesize", rank_and_synthesize) # This node is now LIVE

workflow.set_entry_point("load_patient_profile")
workflow.add_edge("load_patient_profile", "get_contraindications")
workflow.add_edge("get_contraindications", "calculate_bmr")
workflow.add_edge("calculate_bmr", "call_recipe_tool")
workflow.add_edge("call_recipe_tool", "call_workout_tool")
workflow.add_edge("call_workout_tool", "rank_and_synthesize")
workflow.add_edge("rank_and_synthesize", END)

app = workflow.compile()
print("--- Graph Compiled ---")

# -------------------------------------------------
# 6. Run the Agent
# -------------------------------------------------
print("\n--- Running Agent (v2.5) ---")

initial_state = { "user_query": "Find me a healthy meal plan." }
final_state = app.invoke(initial_state)

print("\n--- Final Response ---")
print(final_state["final_recommendation"])