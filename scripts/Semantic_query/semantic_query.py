from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests

app = FastAPI()

# Função para obter embeddings do Nomic local
def get_embedding(context: str, model: str = "nomic-embed-text"):
    """Obtém embeddings usando a API local do Nomic."""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": context}
    )
    if response.status_code == 200:
        return response.json().get("embedding")
    raise Exception(f"[ERROR] Failed to get embedding: {response.text}")

class Consulta(BaseModel):
    consultas: List[str]
    passagens: List[str]

class Resposta(BaseModel):
    pontuacoes: List[List[float]]

@app.post("/calcular-similaridades", response_model=Resposta)
async def calcular_similaridades(dados: Consulta):
    try:
        # Obter embeddings para consultas e passagens
        embeddings_consultas = [get_embedding(c) for c in dados.consultas]
        embeddings_passagens = [get_embedding(p) for p in dados.passagens]

        # Normalizar os embeddings (usando numpy para simplificação)
        import numpy as np
        embeddings_consultas = np.array(embeddings_consultas)
        embeddings_passagens = np.array(embeddings_passagens)

        embeddings_consultas = embeddings_consultas / np.linalg.norm(embeddings_consultas, axis=1, keepdims=True)
        embeddings_passagens = embeddings_passagens / np.linalg.norm(embeddings_passagens, axis=1, keepdims=True)

        # Calcular similaridades
        pontuacoes = np.dot(embeddings_consultas, embeddings_passagens.T).tolist()

        return Resposta(pontuacoes=pontuacoes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
