import chromadb
from sentence_transformers import SentenceTransformer
import os
import re

def smart_chunk_text(text, chunk_size=500, overlap=100):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s

    if current.strip():
        chunks.append(current.strip())

    return chunks

def setup_knowledge_base():
    print("🍳 Setting up Recipe & Nutrition Knowledge Base...")
    
    # Initialize
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Delete old collection if exists
    try:
        client.delete_collection("recipes")
    except:
        pass
    
    collection = client.create_collection("recipes")
    
    # Load all .txt files from data folder
    data_folder = "./data"
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print("📁 Created 'data' folder. Please add recipe files and run again.")
        return
    
    files = [f for f in os.listdir(data_folder) if f.endswith('.txt')]
    
    if not files:
        print("⚠️ No .txt files found in 'data' folder!")
        return
    
    total_chunks = 0
    
    for filename in files:
        filepath = os.path.join(data_folder, filename)
        print(f"📄 Processing: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = smart_chunk_text(text)
        
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                embedding = embedder.encode(chunk).tolist()
                collection.add(
                    ids=[f"{filename}_{i}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{"source": filename, "category": "recipe"}]
                )
                total_chunks += 1
    
    print(f"\n✅ Knowledge base ready!")
    print(f"📊 Total chunks loaded: {total_chunks}")
    print(f"📁 Files processed: {len(files)}")

if __name__ == "__main__":
    setup_knowledge_base()
