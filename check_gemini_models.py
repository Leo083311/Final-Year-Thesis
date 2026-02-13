import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load your API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

print("✅ Checking available models...\n")

# List all models that support generateContent (chat, text, image, etc.)
models = genai.list_models()

for m in models:
    if hasattr(m, "supported_generation_methods") and "generateContent" in m.supported_generation_methods:
        print(f"🧠 {m.name}")
