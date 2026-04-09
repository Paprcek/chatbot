import os
import pickle
import torch
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

app = FastAPI()

class ChatRequest(BaseModel):
    # Změníme query na message, aby to sedělo s tím, co posílá tvůj web
    message: str 

@app.post("/ask")
async def ask_endpoint(request: Request): # Změníme ChatRequest na Request
    try:
        # Tady si "natvrdo" vytáhneme JSON a podíváme se do něj
        data = await request.json()
        print(f"DEBUG: Dorazila data: {data}") # Tohle uvidíš v terminálu!

        # Zkusíme najít text v jakémkoliv běžném klíči
        user_text = data.get('message') or data.get('query') or data.get('text')
        
        if not user_text:
            return {"error": "Nebyl nalezen žádný text dotazu", "received": data}, 400

        # Voláme tvou funkci
        answer, sources = ask_gemini(user_text)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        print(f"❌ Chyba v mozku: {e}")
        return {"error": str(e)}, 500

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
    
    model = genai.GenerativeModel(
        model_name="gemini-3.1-flash-lite-preview",
        system_instruction=(
            "Jsi moudrý, laskavý a civilní průvodce Biblí. Tykáš uživateli. "
            "Pomáháš s debuggingem víry a analýzou toxické komunikace. "
            "Tvým úkolem je propojovat biblickou moudrost s moderním životem a IT terminologií. "
            "Nepoužíváš slang. Tvůj tón je hluboký, ale srozumitelný. "
            "Nepřiznávej uživateli, že ti někdo poslal nějaký externí kontext. Ty ty verše znáš, máš je ve své 'databázi moudrosti'. Nikdy neříkej 'v textech, které jsi mi poslal' nebo 'podle tvých veršů'. Místo toho říkej: 'V Písmu se píše...', 'Biblické texty nám ukazují...' nebo 'Když se podíváme do archivu moudrosti...'."
            "\n\nKONTEXT A SCHOPNOSTI:\n"
            "1. PŘÍBĚHY: Pokud tě uživatel požádá o příběh, převyprávěj ho poutavě a srozumitelně, "
            "jako bys ho vyprávěl u kafe. Zachovej ale věrnost biblickému poselství.\n"
            "2. ANALÝZA: Hledej v textech 'duchovní malware', 'toxické smyčky' a navrhuj 'firewally' pro vnitřní mír.\n"
            "Pokud to pomůže vysvětlit složitý proces (např. nekonečnou smyčku), "
            "vytvoř na konci odpovědi Mermaid diagram v bloku ```mermaid. "
            "Pravidla pro Mermaid: Nepoužívej diakritiku, texty uzlů dávej do uvozovek, např. A[\"Text\"], "
            "nepoužívej speciální znaky jako ?, !, čárky nebo tečky. "
            "V diagramech Mermaid používej jen základní styl graph TD. Texty v uzlech piš bez diakritiky a VŽDY je dávej do uvozovek, např. A[\"Start\"] -> B[\"Cil\"]."
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
    print("🚀 Most pro BIBLEbota se staví na portu 5005...")
    # host="0.0.0.0" zajistí, že na to uvidí i Docker
    uvicorn.run(app, host="0.0.0.0", port=5005)