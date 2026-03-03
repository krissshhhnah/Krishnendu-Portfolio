# --- ELITE BACKEND SERVER (Flask + OpenRouter) ---
import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from flask import render_template

load_dotenv()

# -------------------------------------------------
# APP SETUP
# -------------------------------------------------
app = Flask(__name__)
CORS(app)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set.")

# -------------------------------------------------
# SYSTEM PROMPTS
# -------------------------------------------------
AGENT_SYSTEM_PROMPT = """
You are Kai, the AI assistant for Krishnendu Prasanth’s portfolio.

You are NOT Krishnendu.
You always speak about him in third person.
Never say “I am Krishnendu.”
If asked who you are, say:
"I’m Kai, Krishnendu’s AI assistant."

About Krishnendu:

- AI-focused undergraduate pursuing BCA (Hons) in Artificial Intelligence at Nitte Institute of Professional Education (CGPA: 9.11).
- Software Developer Intern at PixelCode Technologies.
- Builds production-grade AI systems across LLMs, RAG, multi-agent orchestration, review automation, and predictive analytics.
- Hackathon-winning AI supply chain optimization system (~20% stock-out risk reduction).
- Engineered dynamic RAG pipeline with session-isolated vector stores and hallucination control (40% answer relevance boost, 30% latency reduction).
- Built AI-driven Python code evaluation system improving grading consistency by 45%.
- Designed SentinelAI multi-agent engineering review orchestrator reducing manual effort by ~65%.
- Strong in Python, LangChain, Transformers, PyTorch, TensorFlow, Docker, AWS.

Tone Guidelines:
- Professional, sharp, slightly witty.
- Concise (max 2 sentences unless user asks for detail).
- Speak confidently but factually.
- Position him as a systems builder, not just a student.

If asked personal or unrelated questions, redirect politely back to Krishnendu’s work and skills.
"""

ELI5_SYSTEM_PROMPT = """
You are a kindergarten teacher.
Explain the technical project description like a 5-year-old.
Use toy or playground analogies.
Under 30 words only.
"""

# -------------------------------------------------
# HTTP SESSION (FASTER + RETRIES)
# -------------------------------------------------
session = requests.Session()

retries = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)

adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)

# -------------------------------------------------
# MODEL FALLBACK SYSTEM
# -------------------------------------------------
MODELS = [
    "arcee-ai/trinity-large-preview:free",
    "openchat/openchat-7b:free"
]

# -------------------------------------------------
# LLM CALL FUNCTION
# -------------------------------------------------
def call_llm(prompt, system_prompt):

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    for model in MODELS:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        }

        try:
            response = session.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=25
            )

            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]

            elif response.status_code == 429:
                print(f"Rate limited on {model}, trying fallback...")
                continue

            else:
                print("API Error:", response.text)

        except requests.exceptions.Timeout:
            print("Timeout — trying fallback model...")
            continue

        except Exception as e:
            print("LLM Exception:", e)
            continue

    return "⚠️ My AI brain is busy right now. Try again shortly."

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "AI server online"})


@app.route("/agent", methods=["POST"])
def agent_chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    reply = call_llm(user_msg, AGENT_SYSTEM_PROMPT)
    return jsonify({"response": reply})


@app.route("/eli5", methods=["POST"])
def eli5_explain():
    data = request.get_json()
    desc = data.get("description", "")
    reply = call_llm(desc, ELI5_SYSTEM_PROMPT)
    return jsonify({"explanation": reply})


@app.route("/token", methods=["GET"])
def get_token():
    return jsonify({"token": "mock_token_for_demo"})


# -------------------------------------------------
# SERVER START
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🔥 Elite AI Server running on port {port}")
    app.run(host="0.0.0.0", port=10000)



