# ===================== IMPORTS =====================
import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
import os
from dotenv import load_dotenv
import warnings

# ===================== WARNINGS (SUPPRESS TORCH NOISE) =====================
warnings.filterwarnings("ignore")

# ===================== LOAD ENV =====================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ===================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Recipe & Nutrition Chatbot",
    page_icon="🥗",
    layout="wide"
)

# ===================== LOAD CSS =====================
if os.path.exists("styles.css"):
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ===================== EMBEDDINGS ===================
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ===================== CHROMA DB (IN-MEMORY SAFE) ====================
@st.cache_resource
def get_collection():
    # In-memory ChromaDB (no SQLite persistence)
    client = chromadb.Client(
        Settings(
            is_persistent=False,
            anonymized_telemetry=False
        )
    )
    collection = client.get_or_create_collection(name="recipes")
    return collection

collection = get_collection()
embedder = load_embedder()

# ===================== LOAD DATA (Optional) ====================
# Replace this with your actual recipe dataset
# Example format: [{"id": "1", "name": "Oatmeal", "text": "Oatmeal recipe..."}]
my_recipes_list = [
    {"id": "1", "name": "Oatmeal Breakfast", "text": "Oatmeal with fruits and nuts. Serving size: 1 bowl. Calories: 250 kcal."},
    {"id": "2", "name": "Grilled Chicken Salad", "text": "Grilled chicken with mixed greens. Serving size: 1 plate. Calories: 350 kcal."}
]

# Load data into collection if empty
if collection.count() == 0:
    for doc in my_recipes_list:
        collection.add(
            documents=[doc["text"]],
            metadatas=[{"name": doc["name"]}],
            ids=[doc["id"]]
        )

# ===================== LLM FUNCTION ====================
def ask_nutrition_bot(question: str, context: str) -> str:
    system_prompt = f"""
You are an English-only Recipe and Nutrition Chatbot.

STRICT RULES:
1. Communicate ONLY in English.
2. Answer ONLY using the provided context.
3. Do NOT use general knowledge or the internet.
4. Do NOT invent recipes or nutritional values.
5. If information is missing, respond exactly with:
   "The requested information is not available in the current recipe and nutrition dataset."
6. Maintain a clear, professional tone.
7. For medical questions say:
   "This chatbot provides general nutrition information only. Please consult a healthcare professional for medical advice."

CONTEXT FROM DATABASE: {context}

FORMAT:
- Use headings and bullet points
- Include nutritional values if available
- Mention serving size
- Be concise
"""
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.2
        },
        timeout=60
    )

    return response.json()["choices"][0]["message"]["content"]

# ===================== SIDEBAR ======================
with st.sidebar:
    st.markdown("### 🥗 Recipe & Nutrition Bot")
    st.caption("Your personal nutrition assistant")

    st.divider()

    chunk_count = collection.count()
    st.markdown("#### 📊 Knowledge Base")
    st.markdown(
        f"""
        <div class="stats-card">
            <strong>Recipes & Data:</strong> {chunk_count} entries
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("#### 💡 Try Asking About")
    topics = [
        "Weight loss breakfast",
        "High protein meals",
        "Low calorie recipes",
        "Heart healthy diet",
        "Vegetarian protein sources",
        "Daily calorie needs"
    ]
    for topic in topics:
        st.caption(f"• {topic}")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ===================== MAIN CHAT ====================
st.markdown("## 🥗 Recipe & Nutrition Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome Screen
if chunk_count == 0:
    st.warning("⚠️ Knowledge base is empty. Please ensure recipes are loaded.")
elif len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="welcome-card">
            <h2>🥗 Welcome to Recipe & Nutrition Bot</h2>
            <p>Ask about recipes, nutrition values, diet charts, and healthy meals.</p>
            <p style="color:#7cb68a;">
                <strong>Examples:</strong>
                Weight loss breakfast • High protein meals • Low calorie dinner
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Display Messages
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🥗"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Chat Input
prompt = st.chat_input("Ask about recipes, nutrition, or diet plans...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🥗"):
        if chunk_count == 0:
            response = "⚠️ Knowledge base is empty. Please ensure recipes are loaded."
        else:
            query_embedding = embedder.encode(prompt).tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )

            if results["documents"] and results["documents"][0]:
                context = "\n\n---\n\n".join(results["documents"][0])
                response = ask_nutrition_bot(prompt, context)
            else:
                response = "The requested information is not available in the current recipe and nutrition dataset."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
