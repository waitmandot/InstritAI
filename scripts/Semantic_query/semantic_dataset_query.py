from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import requests
from datasets import load_dataset
import json

app = FastAPI()

# Função para obter embeddings do Nomic local
def get_embedding(context: str, model: str = "nomic-embed-text"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": context}
    )
    if response.status_code == 200:
        return response.json().get("embedding")
    raise Exception(f"[ERROR] Failed to get embedding: {response.text}")

# Carregar dataset e pré-processar
print("Carregando e processando dataset...")
dataset = load_dataset("waitmandot/test", split="train")
chunks = [f"{item['title']}: {item['chunk']}" for item in dataset]
print(f"{len(chunks)} chunks encontrados.")

# Gerar embeddings para cada chunk
print("Gerando embeddings para os chunks...")
chunk_embeddings = []
for chunk in chunks:
    embedding = get_embedding(chunk)
    chunk_embeddings.append(embedding)

# Salvar chunks e embeddings em arquivo
with open("chunks_data.json", "w") as f:
    json.dump({"chunks": chunks, "embeddings": chunk_embeddings}, f)
print("Chunks e embeddings salvos em 'chunks_data.json'.")

# API para consulta
class Consulta(BaseModel):
    pergunta: str

class Resposta(BaseModel):
    resultados: List[str]

@app.post("/consulta", response_model=Resposta)
async def consultar(dados: Consulta):
    try:
        # Carregar embeddings salvos
        with open("chunks_data.json", "r") as f:
            data = json.load(f)
        chunks = data["chunks"]
        embeddings = np.array(data["embeddings"])

        # Obter embedding da pergunta
        pergunta_embedding = np.array(get_embedding(dados.pergunta))
        pergunta_embedding = pergunta_embedding / np.linalg.norm(pergunta_embedding)

        # Calcular similaridades
        similarities = np.dot(embeddings, pergunta_embedding)
        top_indices = similarities.argsort()[-3:][::-1]  # Top 3 índices mais similares

        # Retornar os chunks mais similares
        resultados = [chunks[i] for i in top_indices]
        return Resposta(resultados=resultados)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
