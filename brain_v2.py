import os
import pickle
import torch
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# 1. Načtení klíče z .env
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 2. Nastavení cest a modelů
INDEX_PATH = 'data/bible_index.pkl'
EMBED_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

print("🧠 Probouzím lehkého biblického asistenta (API verze)...")

# Lokální část: Vyhledávání (to tvůj server zvládá skvěle)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def get_bible_context(query, top_k=3):
    with open(INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, data['embeddings'])[0]
    top_results = torch.topk(cos_scores, k=top_k)
    
    context = ""
    for idx in top_results.indices:
        context += f"({data['metadata'][idx]}): {data['sentences'][idx]}\n"
    return context

def ask_groq(query):
    # Najdeme verše lokálně
    context = get_bible_context(query)
    
    # Pošleme do Cloudu (tady se děje ta magie)
    print("🚀 Letím pro odpověď do cloudu...")
    
    completion = client.chat.completions.create(
        # Gemma 2 9B je v cloudu neuvěřitelně chytrá a rychlá
        model="llama-3.3-70b-versatile", 
        messages=[
            {
                "role": "system", 
                "content": "Jsi laskavý, laskavý, hluboký a moudrý průvodce Biblí. Tykáš uživateli. "
                           "Odpovídej na základě poskytnutých veršů. Mluv jako přítel, srozumitelně."
                           "Pokud je to užitečné pro vysvětlení (např. u složitých vztahů nebo procesů), vytvoř na konci odpovědi i diagram v Mermaid formátu."
            },
            {
                "role": "user", 
                "content": f"Otázka: {query}\n\nBiblický kontext:\n{context}"
            }
        ],
        temperature=0.7,
        max_tokens=500
    )
    return completion.choices[0].message.content, context

if __name__ == "__main__":
    print("✨ API Asistent připraven. (Ukončíš pomocí Ctrl+C)")
    while True:
        user_query = input("\n👤 Ty: ")
        if not user_query: continue
        
        answer, sources = ask_groq(user_query)
        
        print("\n📖 Odpověď:")
        print("-" * 30)
        print(answer)
        print("-" * 30)
        print(f"\n📚 Verše, které jsem pro tebe našel:\n{sources}")