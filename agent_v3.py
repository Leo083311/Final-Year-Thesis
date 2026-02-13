import os
import json
import re
import random
import psycopg2
import psycopg2.extras
from typing import List, Dict, TypedDict, Any
from dotenv import load_dotenv

import google.generativeai as genai
from langgraph.graph import StateGraph, END

# -----------------------------
# 1. Configuration
# -----------------------------
# # 'dotenv' is used to load hidden passwords (API keys) from a separate file.
# This keeps credentials safe and not visible in the main code.
# initialize Gemini-2.5-flash
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
NEON_DB_URL = "postgresql://neondb_owner:npg_iTDWup6xjB9M@ep-soft-wind-a1ezspim.ap-southeast-1.aws.neon.tech:5432/neondb?sslmode=require"

if not GOOGLE_API_KEY: raise SystemExit("Error: GOOGLE_API_KEY not found.")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# -----------------------------
# 2. State Definition
# -----------------------------
# This 'TypedDict' defines the Schema of your application's memory.
# It ensures data consistency. Every node (function) in your graph 
# reads from this state and writes back to it.
class AgentState(TypedDict):
    # Stores user details (age, weight, allergies) loaded from the file.
    patient_profile: Dict
    
    # Stores rules like "No Sugar" if the user has Diabetes.
    contraindications: Dict
    
    # Stores the calculated calorie numbers.
    caloric_needs: Dict
    
    # Stores the final 7-day meal and workout schedules.
    meal_plan_7_day: List[Dict] 
    workout_plan_7_day: List[Dict]
    
    # Stores a list of any risks found by the Safety Check.
    safety_warnings: Dict[str, str]
    
    # Records any changes the AI made to fix those risks.
    repair_logs: List[str]
    
    # The final text report shown to the user.
    final_report: str

# -----------------------------
# 3. Knowledge Base
# -----------------------------
CONDITION_KEYWORD_MAP = {
    # If the user has Diabetes, block recipes that has the following words.
    "diabetes": ['%cake%', '%cookie%', '%pie%', '%sugar%', '%sweet%', '%dessert%', '%chocolate%', '%ice cream%', '%syrup%', '%honey%', '%candy%', '%pudding%', '%glaze%', '%tart%', '%waffle%', '%pancake%', '%cornflake%', '%breaded%', '%fry%', '%fried%'],
    "hypertension": ['%bacon%', '%sausage%', '%ham%', '%hot dog%', '%salami%', '%cured%', '%soy sauce%', '%teriyaki%', '%miso%', '%pickle%', '%salted%', '%brine%', '%canned%'],
    "blood pressure": ['%bacon%', '%sausage%', '%ham%', '%hot dog%', '%salami%', '%cured%', '%soy sauce%', '%teriyaki%', '%miso%', '%pickle%', '%salted%', '%brine%'],
    "gerd": ['%spicy%', '%chili%', '%curry%', '%jalapeño%', '%hot sauce%', '%buffalo%', '%citrus%', '%lemon%', '%lime%', '%orange%', '%tomato%', '%chocolate%', '%coffee%', '%mint%', '%fried%', '%onion%', '%garlic%'],
    "reflux": ['%spicy%', '%chili%', '%curry%', '%jalapeño%', '%hot sauce%', '%buffalo%', '%citrus%', '%lemon%', '%lime%', '%orange%', '%tomato%', '%chocolate%', '%coffee%', '%mint%', '%fried%', '%onion%', '%garlic%'],
    "kidney": ['%bacon%', '%sausage%', '%ham%', '%liver%', '%organ%', '%bran%', '%chocolate%', '%nuts%', '%banana%', '%dairy%', '%spinach%', '%potato%'],
    "ckd": ['%bacon%', '%sausage%', '%ham%', '%liver%', '%organ%', '%bran%', '%chocolate%', '%nuts%', '%banana%', '%dairy%', '%spinach%', '%potato%'],
    "inflammation": ['%sugar%', '%syrup%', '%bacon%', '%sausage%', '%fried%', '%processed%', '%white bread%'],
    "stress": ['%sugar%', '%syrup%', '%bacon%', '%sausage%', '%fried%', '%processed%', '%caffeine%'],
    "cortisol": ['%sugar%', '%syrup%', '%bacon%', '%sausage%', '%fried%', '%processed%', '%caffeine%'],
    "pcos": ['%sugar%', '%syrup%', '%fried%', '%processed%', '%white bread%', '%pasta%'],
    "gout": ['%liver%', '%kidney%', '%organ%', '%heart%', '%anchovy%', '%sardine%', '%herring%', '%mussel%', '%scallop%', '%beer%', '%yeast%', '%gravy%'],
    "uric acid": ['%liver%', '%kidney%', '%organ%', '%heart%', '%anchovy%', '%sardine%', '%herring%', '%mussel%', '%scallop%', '%beer%', '%yeast%', '%gravy%'],
    "celiac": ['%wheat%', '%barley%', '%rye%', '%bread%', '%pasta%', '%flour%', '%pizza%', '%soy sauce%', '%beer%', '%cake%', '%cookie%', '%biscuit%', '%bagel%', '%crouton%'],
    "gluten": ['%wheat%', '%barley%', '%rye%', '%bread%', '%pasta%', '%flour%', '%pizza%', '%soy sauce%', '%beer%', '%cake%', '%cookie%', '%biscuit%', '%bagel%', '%crouton%'],
    "cholesterol": ['%steak%', '%ribeye%', '%ribs%', '%butter%', '%cream%', '%cheese%', '%bacon%', '%sausage%', '%fried%', '%coconut%', '%palm oil%', '%skin%'],
    "heart disease": ['%steak%', '%ribeye%', '%ribs%', '%butter%', '%cream%', '%cheese%', '%bacon%', '%sausage%', '%fried%', '%coconut%', '%palm oil%', '%skin%']
}

# -----------------------------
# 4. Helpers
# -----------------------------
def get_db_connection(): return psycopg2.connect(NEON_DB_URL)

def expand_allergy_terms(user_allergies):
    # This loop looks at what the user is allergic to.
    # If they say "seafood", it automatically adds specific types like "shrimp" and "crab".
    # This helps the search filter catch every dangerous ingredient.
    """
    V5.9: Expands generic allergy terms into specific ingredient keywords.
    Fixes the 'Seafood vs. Shrimp' blind spot.
    """
    expanded = []
    for a in user_allergies:
        term = a.lower()
        # Always add the original term
        expanded.append(f"%{term}%")
        
        # Expansion
        if "seafood" in term or "shellfish" in term:
            expanded.extend(['%shrimp%', '%prawn%', '%crab%', '%lobster%', '%clam%', '%mussel%', '%oyster%', '%squid%', '%scallop%', '%fish%', '%salmon%', '%tuna%', '%cod%', '%tilapia%', '%halibut%', '%conch%', '%octopus%', '%bass%', '%trout%'])
        elif "gluten" in term or "wheat" in term:
            expanded.extend(['%wheat%', '%barley%', '%rye%', '%bread%', '%pasta%', '%flour%', '%soy sauce%', '%seitan%', '%bulgur%', '%couscous%', '%farro%'])
        elif "dairy" in term or "lactose" in term or "milk" in term:
            expanded.extend(['%milk%', '%cheese%', '%cream%', '%yogurt%', '%butter%', '%whey%', '%casein%', '%ghee%', '%custard%', '%gelato%'])
        elif "nut" in term:
            expanded.extend(['%almond%', '%cashew%', '%walnut%', '%pecan%', '%pistachio%', '%macadamia%', '%hazelnut%', '%peanut%'])
        elif "egg" in term:
            expanded.extend(['%egg%', '%mayonnaise%', '%meringue%', '%custard%'])
            
    return list(set(expanded))

def calculate_maintenance_calories(profile: Dict) -> Dict:
    # Uses weight, height, sex andage to find the BMR (Basal Metabolic Rate).
    # Multiplies BMR by activity level to find total daily energy needs.
    # Adjusts the total based on the goal (subtracts 500 for weight loss).
    weight = profile.get('weight_kg', 70)
    height = profile.get('height_cm', 175)
    age = profile.get('age', 25)
    sex = profile.get('sex', 'male').lower()
    activity = profile.get('activity_level', 'moderate') 
     # This is where the logic would swap 'loss' for 'maintenance' 
    # if the weight_kg is too low.
    goal = profile.get('primary_goal', 'maintenance').lower()
   

    if sex == 'male': bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else: bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161
    
    multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}
    tdee = bmr * multipliers.get(activity, 1.375)

    target_calories = tdee
    if "loss" in goal: target_calories = tdee - 500
    elif "gain" in goal: target_calories = tdee + 300

    return {"bmr": int(bmr), "tdee": int(tdee), "target_calories": int(target_calories), "goal_type": goal}

def calculate_macros(calories: int, goal: str, safety_rules: Dict, profile: Dict) -> Dict:
    # 1. ESTABLISH BASELINE RATIOS
    # Default "Balanced" diet: 40% Carbs, 30% Protein, 30% Fat.
    r_c, r_p, r_f = 0.4, 0.3, 0.3
    # Adjusts ratios based on the user's goal.
    # If "Gain": Increases Carbs/Protein slightly to fuel workouts.
    # If "Loss": Increases Protein to 40% to help keep the user full and save muscle.
    if "gain" in goal: r_c, r_p, r_f = 0.4, 0.35, 0.25
    elif "loss" in goal: r_c, r_p, r_f = 0.3, 0.4, 0.3

# 2. SAFETY CAP #1: GENERAL HEALTH
    # calculate a "Maximum Safe Protein Limit" based on body weight (2.5g per kg).
    # This prevents the system from giving a user 300g of protein, which is unsafe and not logical
    weight = profile.get('weight_kg', 70)
    max_protein_g = weight * 2.5 
    
    # Calculate protein grams
    calculated_protein_g = int((calories * r_p) / 4)
    # If the calculated protein is too high, we force it down to the Max Limit.
    if calculated_protein_g > max_protein_g:
        final_protein_g = int(max_protein_g)
        # take the "leftover" calories and give them to Carbs and Fat.
        remaining_cals = calories - (final_protein_g * 4)
        final_carbs_g = int((remaining_cals * 0.6) / 4)
        final_fat_g = int((remaining_cals * 0.4) / 9)
        #if not too high keep it 
    else:
        final_protein_g = calculated_protein_g
        final_carbs_g = int((calories * r_c) / 4)
        final_fat_g = int((calories * r_f) / 9)

# Check if user has medical conditions that require special protein limits
    avoid_goals = safety_rules.get("avoid_recipe_goals", [])
    if "high_protein" in avoid_goals:
        # limit protein intake to 1.0g per kg of body weight
        limit_protein_g = weight * 1.0
        # If the final protein exceeds this limit, adjust it down again
        if final_protein_g > limit_protein_g:
            print(f"⚠️ SAFETY TRIGGER: Restricting Protein to {int(limit_protein_g)}g for CKD.")
            final_protein_g = int(limit_protein_g)

            # take the "leftover" calories and give them to Carbs and Fat.
            remaining_cals = calories - (final_protein_g * 4)
            final_carbs_g = int((remaining_cals * 0.65) / 4)
            final_fat_g = int((remaining_cals * 0.35) / 9)
    
    return {"protein_g": final_protein_g, "carbs_g": final_carbs_g, "fat_g": final_fat_g}

# Generates a list of banned keywords based on the user's medical conditions.
def get_banned_keywords(conditions: List[str]) -> List[str]:
    banned = []
    for cond in conditions:
        for key, keywords in CONDITION_KEYWORD_MAP.items():
            if key in cond.lower():
                banned.extend(keywords)
    return list(set(banned))

# Searches the database for meal options that fit the calorie target and avoid allergens/banned keywords.
def get_substitute_meal_options(target_calories, profile, specific_exclusion=None, limit=3):
    """
    V5.9 FIX: Strict Parameter Ordering to ensure Safety Filters work.
    """
    # Connects to the database.
    conn = get_db_connection()
    
    # Sets a calorie range +/- 10-15% based on target calories.
    if target_calories < 500:
        min_cal, max_cal = target_calories * 0.8, target_calories * 1.1
    else:
        min_cal, max_cal = target_calories * 0.85, target_calories * 1.15
        
    raw_allergies = profile.get('allergies', [])
    # V5.9: Use the new expander
    allergy_patterns = expand_allergy_terms(raw_allergies)
    
    conditions = profile.get('conditions', [])
    banned_keywords = get_banned_keywords(conditions)

    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
    # Builds a search command (SQL Query).
    # 1. Filters recipes to match the required calorie range.
    # 2. Filters out recipes containing allergens or banned keywords.
        query = """
            SELECT metadata_->>'title' as title, 
                   metadata_->>'instructions' as instructions,
                   metadata_->>'cleaned_ingredients_str' as ingredients,
                   calories, protein_g 
            FROM data_recipes_data 
            WHERE nutrition_labeled = TRUE 
            AND calories BETWEEN %s AND %s
            AND protein_g > 15
        """
        # Executes the search and returns a random valid recipe.
        params = [min_cal, max_cal]
        
        # Applies allergy filters if any exist.
        if allergy_patterns:
            query += """ AND NOT (
                LOWER(metadata_->>'cleaned_ingredients_str') ILIKE ANY(%s) 
                OR LOWER(metadata_->>'title') ILIKE ANY(%s)
            )"""
            params.append(allergy_patterns)
            params.append(allergy_patterns) # Must append twice because we use %s twice
            print(f"   [Debug] Filtering Allergens (Ingredients + Title): {allergy_patterns}")
        
        # Applies banned keyword filters if any exist.
        if banned_keywords:
            query += " AND NOT (LOWER(metadata_->>'title') ILIKE ANY(%s))"
            params.append(banned_keywords)

        # Applies specific exclusion if provided.   
        if specific_exclusion:
            query += " AND metadata_->>'title' != %s"
            params.append(specific_exclusion)
        
        query += f" ORDER BY RANDOM() LIMIT %s;"
        params.append(limit)
        
        try:
            cur.execute(query, params)
            rows = cur.fetchall()
        except Exception as e:
            print(f"   [Error] SQL Replacement Failed: {e}")
            rows = []
            
    conn.close()
    return [dict(row) for row in rows]

# Searches the database for workout options similar to the previous suggested exercise.
def get_substitute_workout_options(old_exercise, limit=3):
    conn = get_db_connection()
    mechanic = old_exercise.get('mechanic', 'compound')
    force = old_exercise.get('force_type', 'push')
    tier = old_exercise.get('tier', 'primary')
    old_title = old_exercise.get('title', '')

# Builds and runs the query, psychopg2 will handle SQL injection safety by remembering the order of parameters.
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        query = """
            SELECT metadata_->>'title' as title, 
                   metadata_->>'description' as description,
                   mechanic, force_type, tier, duration_min 
            FROM data_workout_data
            WHERE details_labeled = TRUE
            AND mechanic = %s
            AND force_type = %s
            AND tier = %s
            AND metadata_->>'title' != %s
            ORDER BY RANDOM() LIMIT %s;
        """
        cur.execute(query, (mechanic, force, tier, old_title, limit))
        rows = cur.fetchall()
    conn.close()
    
    options = []
    for row in rows:
        ex = dict(row)
        ex['reps_display'] = "3 Sets x 10 Reps"
        options.append(ex)
    return options

# -----------------------------
# 5. Nodes
# -----------------------------
# Opens the medical_report.json file to get user data.
def load_patient_profile(state: AgentState):
    print("\n--- Node 1: Loading Profile ---")
    try:
        with open("medical_report.json", "r") as f: profile = json.load(f)
        print(f"✅ Loaded: {profile.get('name')}")
        return {"patient_profile": profile}
    except FileNotFoundError: return {"patient_profile": None}

# Checks the user's medical conditions against the knowledge base.
def get_medical_rules(state: AgentState):
    print("--- Node 2: Checking Contraindications ---")
    profile = state['patient_profile']
    # If no profile is found, return empty.
    if not profile: return {}
    conditions = profile.get('conditions', [])
    # Queries the database for rules related to each condition.
    rules = {"avoid_recipe_goals": [], "avoid_workout_intensity": [], "avoid_mechanic": []}
    # Connects to the database.
    conn = get_db_connection()
    # Creates a cursor to execute SQL queries.
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # For each condition, fetches the related rules.
        for cond in conditions:
            cur.execute("SELECT * FROM data_medical_knowledge WHERE condition_name = %s", (cond,))
            row = cur.fetchone()
            # If rules are found, they are added to the list.
            if row:
                rules["avoid_recipe_goals"].extend(row.get('avoid_recipe_goals', []))
                rules["avoid_workout_intensity"].extend(row.get('avoid_workout_intensity', []))
    conn.close()
    return {"contraindications": rules}

# Runs the calorie calculation function above.
def run_math_engine(state: AgentState):
    print("--- Node 3: Running Math Engine ---")
    profile = state['patient_profile']
    rules = state['contraindications']
    cal_data = calculate_maintenance_calories(profile)
    macros = calculate_macros(cal_data['target_calories'], cal_data['goal_type'], rules, profile)
    print(f"📊 Targets: {cal_data['target_calories']} kcal | P: {macros['protein_g']}g")
    return {"caloric_needs": cal_data, "macro_split": macros}

# Loops through 7 days, searches the database for breakfast, lunch, and dinner, and applies filters to ensure no allergens are included.
def generate_meal_plan(state: AgentState):
    print("--- Node 4: Generating 7-Day Meal Plan (V5.9 Smart Safety) ---")
    conn = get_db_connection()
    meal_plan = []
    # Sets calorie targets per meal.
    target_cal_per_meal = state['caloric_needs']['target_calories'] / 3
    total_target = state['caloric_needs']['target_calories']

    # Sets calorie ranges with tighter bounds for lower calorie needs.
    if total_target < 1600:
        min_cal = target_cal_per_meal * 0.75
        max_cal = target_cal_per_meal * 1.05 
    # tighter range to avoid large swings
    else:
        min_cal = target_cal_per_meal * 0.8
        max_cal = target_cal_per_meal * 1.2
    
    profile = state['patient_profile']
    conditions = profile.get('conditions', [])
    raw_allergies = profile.get('allergies', [])
    
    # V5.9: Use the new expander
    allergy_patterns = expand_allergy_terms(raw_allergies)
    
    banned_keywords = get_banned_keywords(conditions)
    # Builds and runs the query, psychopg2 will handle SQL injection safety by remembering the order of parameters.
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        base_query = """
            SELECT metadata_->>'title' as title, 
                   metadata_->>'instructions' as instructions,
                   metadata_->>'cleaned_ingredients_str' as ingredients,
                   calories, protein_g, fat_g, carbs_g 
            FROM data_recipes_data 
            WHERE nutrition_labeled = TRUE 
            AND calories BETWEEN %s AND %s
            AND protein_g > 15
        """
        params_list = [min_cal, max_cal]
        filters = ""
        
        # Check if the user has allergies (e.g., "Nut Allergy" -> Expanded to [%almond%, %walnut%...])
        if allergy_patterns:
            # Add a strict rule to the SQL query:
            # "AND NOT" means exclude.
            # check BOTH the 'ingredients' AND the 'title'. 
            # If "peanut" appears in EITHER, the recipe is gone.
            filters += """ AND NOT (
                LOWER(metadata_->>'cleaned_ingredients_str') ILIKE ANY(%s) 
                OR LOWER(metadata_->>'title') ILIKE ANY(%s)
            )"""

            # Append 'allergy_patterns' TWICE because we have two placeholders (%s) above.
            # The database needs the list once for the ingredients check and once for the title check.
            params_list.append(allergy_patterns)
            params_list.append(allergy_patterns) # Append twice
            print(f"🛡️ Safety: Filtering Allergens (Ingredients + Title): {len(allergy_patterns)} terms")

        # Check if the user has medical bans (e.g., Diabetes -> No "Cake")
        if banned_keywords:
            filters += " AND NOT (LOWER(metadata_->>'title') ILIKE ANY(%s))"
            params_list.append(banned_keywords)
            print(f"🛡️ Safety: Blocking {len(banned_keywords)} risky keywords.")

        # A manual list of words that define "Morning Food", could be expanded in the future reaserch. 
        bf_keywords = ['%egg%', '%oat%', '%pancake%', '%waffle%', '%yogurt%', '%breakfast%', '%toast%', '%smoothie%', '%cereal%', '%porridge%', '%granola%', '%fruit%', '%scramble%', '%omelet%', '%benedict%', '%hash%']
        
        # Take the Base Query + Safety Filters + AND the Breakfast Keywords.
        # This says: "Find me a safe meal that ALSO has 'egg' or 'oat' in the title."
        bf_query = base_query + filters + " AND (LOWER(metadata_->>'title') ILIKE ANY(%s)) ORDER BY RANDOM() LIMIT 20;"
        bf_params = params_list + [bf_keywords]
        # Execute and get 20 random breakfast options.
        cur.execute(bf_query, bf_params)
        bf_rows = [dict(r) for r in cur.fetchall()]
        
        # Base Query + Safety Filters.
        # Usually lunch and dinner are almost the same so I did not add the breakfast restrictions here
        # ORDER BY RANDOM() ensures the user gets different meals every time they generate a plan.
        main_query = base_query + filters + " ORDER BY RANDOM() LIMIT 100;"
        cur.execute(main_query, params_list)
        main_rows = [dict(r) for r in cur.fetchall()]
        
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        bf_idx, main_idx = 0, 0
        
        for day in days:
            daily_meals = []
            if bf_idx < len(bf_rows):
                daily_meals.append(bf_rows[bf_idx]); bf_idx += 1
            else: daily_meals.append({"title": "Oatmeal (Fallback)", "calories": int(target_cal_per_meal), "instructions": "Cook oats."})
            
            for _ in range(2):
                if main_idx < len(main_rows):
                    daily_meals.append(main_rows[main_idx]); main_idx += 1
                else: daily_meals.append({"title": "Chicken Salad (Fallback)", "calories": int(target_cal_per_meal), "instructions": "Mix greens."})

            meal_plan.append({"day": day, "meals": daily_meals, "total_calories": sum(m.get('calories',0) for m in daily_meals)})

    conn.close()
    return {"meal_plan_7_day": meal_plan}

# Checks if the user is sedentary (inactive) or active.
# If sedentary: Searches for "Mobility" and "Stability" exercises (safer).
# If active: Searches for "Compound" exercises like squats (harder).
def generate_workout_plan(state: AgentState):
    print("--- Node 5: Generating Adaptive Workout Split ---")
    workout_plan = []
    conn = get_db_connection()
    
    profile = state['patient_profile']
    is_sedentary = profile.get('activity_level', '').lower() == 'sedentary' or profile.get('activity_level', '').lower() == 'none_active'
    
    if is_sedentary:
        # For beginners and less active users, the exercises are easier and safer
        print("🛡️ Safety: Applying Sedentary/Beginner Filters")
        blueprint = [
            {"slot": "Mobility/Warmup", "force": "static", "muscle": "quadriceps", "mechanic": "isometric"},
            {"slot": "Legs/Supported", "force": "push", "muscle": "quadriceps", "mechanic": "isolation"},
            {"slot": "Push/Wall_or_Knee", "force": "push", "muscle": "chest", "mechanic": "compound"},
            {"slot": "Pull/Band_or_Machine", "force": "pull", "muscle": "back", "mechanic": "compound"},
            {"slot": "Core/Stability", "force": "static", "muscle": "abdominals", "mechanic": "isometric"}
        ]
    else:
        # For active users, the system will give standard muscle building exercises like Push/Pull/Legs
        blueprint = [
            {"slot": "Legs/Compound", "force": "push", "muscle": "quadriceps", "mechanic": "compound"},
            {"slot": "Push/Compound", "force": "push", "muscle": "chest", "mechanic": "compound"},
            {"slot": "Pull/Compound", "force": "pull", "muscle": "back", "mechanic": "compound"},
            {"slot": "Shoulders", "force": "push", "muscle": "shoulders", "mechanic": "isolation"},
            {"slot": "Arms/Core", "force": "pull", "muscle": "biceps", "mechanic": "isolation"}
        ]
    
    days = ["Monday", "Wednesday", "Friday"]
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        for day in days:
            daily_exercises = []
            for item in blueprint:
                query = """
                    SELECT metadata_->>'title' as title, 
                           metadata_->>'description' as description,
                           mechanic, force_type, tier, duration_min 
                    FROM data_workout_data
                    WHERE details_labeled = TRUE
                    AND force_type = %s
                    AND mechanic = %s
                    AND metadata_->>'target_muscle' ILIKE %s
                """
                params = [item['force'], item['mechanic'], f"%{item['muscle']}%"]
                
                # The system explicitly BLOCK high-impact or dangerous moves.
                if is_sedentary:
                    query += " AND NOT (LOWER(metadata_->>'title') ILIKE ANY(%s))"
                    params.append(['%jump%', '%plyo%', '%sprint%', '%burpee%', '%barbell%', '%weighted%'])
                
                query += " ORDER BY RANDOM() LIMIT 1;"
                cur.execute(query, params)
                row = cur.fetchone()
                
                if row:
                    ex = dict(row)
                    ex['slot_focus'] = item['slot']
                    title_lower = ex['title'].lower()
                    # Adapts the instructions based on the exercise type and user level.
                    if any(x in title_lower for x in ['walk', 'run', 'treadmill', 'cycle', 'row', 'elliptical']):
                        # If it's cardio, show "15 mins" instead of "10 reps".
                        ex['reps_display'] = f"{ex.get('duration_min', 15)} mins"
                    elif is_sedentary: ex['reps_display'] = "2 Sets x 10 Reps (Light)"
                    # Beginners get lighter volume (2 sets).
                    else: ex['reps_display'] = "3 Sets x 8 Reps"
                    # Active users get standard volume (3 sets).
                    daily_exercises.append(ex)
                else:
                    daily_exercises.append({"title": f"Standard {item['slot']} Movement", "reps_display": "2 Sets x 10 Reps"})

            workout_plan.append({"day": day, "focus": "Full Body", "exercises": daily_exercises})

    conn.close()
    return {"workout_plan_7_day": workout_plan}

def safety_auditor(state: AgentState):
    # Sends the generated meal list to the AI (Gemini).
    # Asks the AI to act as a "Safety Auditor" and find any conflicts 
    # between the meals and the patient's conditions.
    print("--- Node 6: Running Safety Auditor ---")
    meal_titles = [m['title'] for day in state['meal_plan_7_day'] for m in day['meals']]
    prompt = f"""
    You are a Clinical Safety Auditor.
    PATIENT CONDITIONS: {state['patient_profile'].get('conditions')}
    ALLERGIES: {state['patient_profile'].get('allergies')}
    MEAL LIST: {json.dumps(meal_titles)}
    TASK: Identify UNSAFE meals.
    Return ONLY a JSON object: {{ "Meal Title": "Specific Risk Warning" }}
    IMPORTANT: Do NOT flag generic allergens (like shrimp/nuts) unless the patient EXPLICITLY lists them in the ALLERGIES list above.
    """
    try:
        response = gemini_model.generate_content(prompt)
        text = response.text
        json_str = re.search(r"\{.*\}", text, re.DOTALL).group()
        warnings = json.loads(json_str)
        print(f"🛡️ Auditor found {len(warnings)} risks.")
        return {"safety_warnings": warnings}
    # In case the AI fails or returns invalid JSON, we catch the error here.
    except Exception as e:
        print(f"⚠️ Auditor Error: {e}")
        return {"safety_warnings": {}}

# If the Auditor found warnings, this function runs.
# It takes the bad meal out and searches the database again for a safe replacement.
def auto_repair_node(state: AgentState):
    print("--- Node 6.5: Running Auto-Repair ---")
    # Gets the list of flagged meals from the Auditor.
    warnings = state.get('safety_warnings', {})
    meal_plan = state['meal_plan_7_day']
    profile = state['patient_profile']
    repair_logs = []
    
    # If no warnings, skip repair.
    if not warnings:
        print("✅ No repairs needed.")
        return {"repair_logs": [], "meal_plan_7_day": meal_plan}

    # Loops through the meal plan and replaces flagged meals.
    print(f"🔧 Attempting to repair {len(warnings)} meals...")
    for day in meal_plan:
        for i, meal in enumerate(day['meals']):
            # If the meal is flagged, search for a safe replacement.
            if meal['title'] in warnings:
                risk = warnings[meal['title']]
                print(f"   - Replacing '{meal['title']}' (Risk: {risk})")
                # Call the Database Search function again.
                # Crucial Parameter: specific_exclusion=meal['title'], this ensures we don't accidentally pick the same bad meal again.
                options = get_substitute_meal_options(meal['calories'], profile, specific_exclusion=meal['title'], limit=1)
                # Updates the plan with the new, safe meal.
                if options:
                    day['meals'][i] = options[0] 
                    # Log the change so we can tell the user later.
                    repair_logs.append(f"Auto-Replaced '{meal['title']}' with '{options[0]['title']}' (Reason: {risk})")
                else: repair_logs.append(f"Could not find substitute for '{meal['title']}'")

    print(f"✅ Repairs Complete. {len(repair_logs)} swaps performed.")
    return {"meal_plan_7_day": meal_plan, "repair_logs": repair_logs}

# Final synthesis node that creates the comprehensive report.
def synthesize_final_report(state: AgentState):
    print("--- Node 7: Gemini Synthesis ---")
    repairs = state.get('repair_logs', [])
    prompt = f"""
    You are an expert Medical Nutritionist & Fitness Planner.
    PATIENT PROFILE: {json.dumps(state['patient_profile'])}
    THE FINAL MEAL PLAN: {json.dumps(state['meal_plan_7_day'], indent=2)}
    AUTO-REPAIR LOGS: {json.dumps(repairs, indent=2)}
    WORKOUTS: {json.dumps(state['workout_plan_7_day'], indent=2)}
    
    TASK: Generate a comprehensive, patient-friendly Medical Report.
    
    SECTION 1: EXECUTIVE SUMMARY
    - State the Patient's Age, Weight, Height, BMR, and TDEE exactly as calculated.
    - State their primary Goal (e.g., Weight Loss, Muscle Gain).

    SECTION 2: DIETARY STRATEGY & ANALYSIS
    - Write a paragraph explaining the "Why" behind this plan.
    - Analyze the patient's conditions (e.g., CKD, Diabetes) and Allergies.
    - Explain the nutritional strategy used to manage them (e.g., "To support your Chronic Kidney Disease, we have prioritized proteins that are lower in phosphorus...").
    - Mention the "Safety First" approach regarding their allergies.
    - This section should be educational and reassuring.

    SECTION 3: 7-DAY MEAL SCHEDULE
    - List the meals exactly as they appear in the "THE FINAL MEAL PLAN".
    - Use this clean format:
      **[Day Name]**
      * Breakfast: [Meal Name] (Cal: [x], P: [x]g)
      * Lunch: ...
      * Dinner: ...
      * Daily Total: [Sum] kcal
    - CRITICAL: Do NOT put warning labels here. Keep this section clean and readable.

    SECTION 4: WORKOUT ROUTINE
    - List the exercises day by day.

    SECTION 5: SAFETY AUDIT LOG
    - Transparency Section: List the specific adjustments the AI Agent made.
    - Use this format: "• Replaced [Old Meal] with [New Meal] because [Reason]."
    """
    response = gemini_model.generate_content(prompt)
    return {"final_report": response.text}

# -----------------------------
# 5. Graph
# -----------------------------
# Creates the workflow.
workflow = StateGraph(AgentState)
# Adds the steps (nodes) to the graph.
workflow.add_node("load_patient_profile", load_patient_profile)
workflow.add_node("get_medical_rules", get_medical_rules)
workflow.add_node("run_math_engine", run_math_engine)
workflow.add_node("generate_meal_plan", generate_meal_plan)
workflow.add_node("generate_workout_plan", generate_workout_plan)
workflow.add_node("safety_auditor", safety_auditor) 
workflow.add_node("auto_repair_node", auto_repair_node) 
workflow.add_node("synthesize_final_report", synthesize_final_report)

# Defines the path: Profile -> Math -> Meal Plan -> Workout -> Audit -> Repair -> Report
workflow.set_entry_point("load_patient_profile")
workflow.add_edge("load_patient_profile", "get_medical_rules")
workflow.add_edge("get_medical_rules", "run_math_engine")
workflow.add_edge("run_math_engine", "generate_meal_plan")
workflow.add_edge("generate_meal_plan", "generate_workout_plan")
workflow.add_edge("generate_workout_plan", "safety_auditor")
workflow.add_edge("safety_auditor", "auto_repair_node") 
workflow.add_edge("auto_repair_node", "synthesize_final_report") 
workflow.add_edge("synthesize_final_report", END)

app = workflow.compile()

# -----------------------------
# 6. Interactive Execution & State Management
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting V5.9 Planner Agent (Thesaurus Safety Enabled)...")
    # This single line triggers the entire LangGraph workflow.
    # It runs every node (Profile -> Math -> Meal -> Workout -> Audit -> Repair).
    result = app.invoke({})
    
    # Extracts the final meal and workout plans from the result.
    final_meal_plan = result.get('meal_plan_7_day', [])
    final_workout_plan = result.get('workout_plan_7_day', [])
    
    print("\n" + "="*50 + "\nFINAL REPORT GENERATED\n" + "="*50)
    print(result.get("final_report", "No report generated."))
    
    def refresh_lookup_tables():
        # Creates a quick reference map: "name" -> "data object"
        m_lookup = {}
        for day in final_meal_plan:
            for m in day['meals']:
                m_lookup[m['title'].lower()] = m
        w_lookup = {}
        for day in final_workout_plan:
            for ex in day['exercises']:
                w_lookup[ex['title'].lower()] = ex
        return m_lookup, w_lookup

    # Refresh lookup tables so the system knows what meals exist.
    all_meals, all_workouts = refresh_lookup_tables()
    
    print("\n" + "="*50)
    print("👩‍🍳 INTERACTIVE ASSISTANT (V5.9)")
    print("="*50)
    print("Available Commands:")
    print("1. [Recipe Name] -> View details.")
    print("2. [Exercise Name] -> View details.")
    print("3. replace [Item Name] -> Choose a substitute.")
    print("4. review -> Show the FULL updated 7-day plan.")
    print("5. exit -> Quit.")
    
    # Main interactive loop
    while True:
        user_input = input("\n> ").strip().lower()
        if user_input in ['exit', 'quit']: break
        
        if user_input == 'review':
            print("\n" + "="*20 + " UPDATED MEAL PLAN " + "="*20)
            for day in final_meal_plan:
                print(f"\n📅 {day['day']} (Total: {day['total_calories']} kcal)")
                for slot, meal in zip(["Breakfast", "Lunch", "Dinner"], day['meals']):
                    print(f"   - {slot}: {meal['title']} ({meal['calories']} kcal)")
            
            print("\n" + "="*20 + " UPDATED WORKOUT PLAN " + "="*20)
            for day in final_workout_plan:
                print(f"\n💪 {day['day']} ({day['focus']})")
                for ex in day['exercises']:
                    print(f"   - {ex['title']} ({ex.get('reps_display', 'N/A')})")
            continue

        # Checks if the user typed "replace [something]"
        is_replace = user_input.startswith("replace ")
        target_name = user_input.replace("replace ", "").strip() if is_replace else user_input
        
        found_meal = None
        for title, data in all_meals.items():
            # Uses the Lookup Table built earlier to find the meal object instantly.
            if target_name in title: found_meal = data; break
        
        found_workout = None
        for title, data in all_workouts.items():
            if target_name in title: found_workout = data; break
                
        if found_meal:
            if is_replace:
                print(f"🔍 Finding alternatives for Meal: {found_meal['title']}...")
                # Calls the Database Search function again to find specific substitutes, the maximum option is 3
                # Notice specific_exclusion=found_meal['title'] prevents getting the same meal back.
                options = get_substitute_meal_options(found_meal['calories'], result['patient_profile'], specific_exclusion=found_meal['title'], limit=3)
                
                if options:
                    print(f"\n✅ Select a Replacement:")
                    for i, opt in enumerate(options):
                        print(f"   {i+1}. {opt['title']} ({opt['calories']} kcal | P: {opt['protein_g']}g)")
                    
                    try:
                        choice = input("\nType 1, 2, or 3 (or 'c' to cancel): ").strip()
                        # Asks the user to pick one of the options.
                        if choice in ['1', '2', '3']:
                            selected = options[int(choice)-1]
                            # Update the dictionary in place.
                            # Because 'found_meal' points to the exact object inside 'final_meal_plan',
                            # Instead of deleting the old meal and inserting a new one (which messes up the list order),
                            # I simply erase the contents of the old meal and paste the new data into the same container.
                            # This keeps the 7-Day Schedule structure intact.
                            found_meal.clear()
                            found_meal.update(selected)
                            # Re-build the index so the new meal is searchable.
                            all_meals, all_workouts = refresh_lookup_tables()
                            print(f"   (Plan Updated. Type 'review' to see the full schedule.)")
                        else:
                            print("   (Cancelled.)")
                    except Exception:
                        print("   (Invalid input. Cancelled.)")
                else: print("⚠️ No safe substitute found.")
            else:
                print(f"\n🥘 RECIPE: {found_meal['title']}")
                print(f"🔥 Calories: {found_meal['calories']} | P: {found_meal['protein_g']}g")
                print(f"🥕 Ingredients: {found_meal.get('ingredients', 'N/A')[:150]}...")
                print("-" * 20)
                print(found_meal.get('instructions', 'No instructions.'))

        # Workout Details
        elif found_workout:
            if is_replace:
                print(f"🔍 Finding alternatives for Exercise: {found_workout['title']}...")
                options = get_substitute_workout_options(found_workout, limit=3)
                
                # If options are found, display them to the user.
                if options:
                    print(f"\n✅ Select a Replacement:")
                    for i, opt in enumerate(options):
                        print(f"   {i+1}. {opt['title']}")
                    
                    try:
                        choice = input("\nType 1, 2, or 3 (or 'c' to cancel): ").strip()
                        if choice in ['1', '2', '3']:
                            selected = options[int(choice)-1]
                            # same logic as meals, update in place
                            found_workout.clear()
                            found_workout.update(selected)
                            all_meals, all_workouts = refresh_lookup_tables()
                            print(f"   (Plan Updated. Type 'review' to see the full schedule.)")
                        else:
                            print("   (Cancelled.)")
                    except Exception:
                        print("   (Invalid input. Cancelled.)")
                else: print("⚠️ No similar exercise found.")
            else:
                print(f"\n💪 EXERCISE: {found_workout['title']}")
                print(f"📝 Description: {found_workout.get('description', 'N/A')[:150]}...")
                print("-" * 20)
                print("Tip: Keep form strict.")
        else: print("❌ Item not found. Check spelling.")