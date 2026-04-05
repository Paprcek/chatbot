import os
import pickle
import torch
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util

# 1. Načtení klíčů a konfigurace Google API
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# 2. Cesty k datům (tvoje naindexovaná Bible)
INDEX_PATH = 'data/bible_index.pkl'
EMBED_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

print("🧠 Probouzím moudrého asistenta Gemini...")

# Načtení modelu pro vyhledávání (běží u tebe lokálně, je lehký)
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

def get_bible_context(query, top_k=3):
    """Najde v tvém indexu nejrelevantnější verše."""
    with open(INDEX_PATH, 'rb') as f:
        data = pickle.load(f)
    
    query_embedding = embed_model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, data['embeddings'])[0]
    top_results = torch.topk(cos_scores, k=top_k)
    
    context = ""
    for idx in top_results.indices:
        context += f"({data['metadata'][idx]}): {data['sentences'][idx]}\n"
    return context

def ask_gemini(query):
    """Pošle dotaz i s kontextem do Google cloudu."""
    context = get_bible_context(query)
    
    # Nastavení "osobnosti" bota
    # Používáme gemini-1.5-flash (je rychlý a zdarma)
    model = genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite-preview",
        system_instruction=(
            "Jsi moudrý, laskavý a civilní průvodce Biblí. Tykáš uživateli. "
            "Pomáháš s debuggingem víry a analýzou toxické komunikace. "
            "Nepoužíváš slang. Tvůj tón je hluboký, ale srozumitelný. "
            "Pokud to pomůže vysvětlit složitý proces (např. nekonečnou smyčku), "
            "vytvoř na konci odpovědi Mermaid diagram v bloku ```mermaid."
        )
    )
    
    full_prompt = f"""
    Uživatel se ptá: {query}
    
    Zde jsou relevantní biblické verše pro tvou odpověď:
    {context}
    
    Odpověz moudře a srozumitelně.
    """
    
    response = model.generate_content(full_prompt)
    return response.text, context

if __name__ == "__main__":
    print("✨ Gemini je připraven na start! (Ctrl+C pro konec)")
    while True:
        try:
            user_query = input("\n👤 Ty: ")
            if not user_query.strip():
                continue
                
            print("🚀 Letím pro moudrost do cloudu...")
            answer, sources = ask_gemini(user_query)
            
            print("\n📖 Odpověď:\n" + "-"*40)
            print(answer)
            print("-"*40)
            print(f"\n📚 Verše, které jsem použil jako základ:\n{sources}")
            
        except KeyboardInterrupt:
            print("\n👋 Raketoplán přistává. Hezký den!")
            break
        except Exception as e:
            print(f"❌ Ups, v systému je chyba: {e}")