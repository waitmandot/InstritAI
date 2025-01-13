import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datasets import load_dataset
import json

app = FastAPI()

# Caminho do arquivo de embeddings
embedding_file_path = "documents_embeddings.json"

# Função para obter embeddings do Nomic local
def get_embedding(context: str, model: str = "nomic-embed-text"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": context}
    )
    if response.status_code == 200:
        return response.json().get("embedding")
    raise Exception(f"[ERROR] Failed to get embedding: {response.text}")

# Função para carregar ou gerar embeddings
def carregar_ou_gerar_embeddings():
    if os.path.exists(embedding_file_path):
        print(f"Arquivo '{embedding_file_path}' encontrado. Carregando documentos e embeddings existentes...")
        with open(embedding_file_path, "r") as f:
            data = json.load(f)
            return data["documents"], data["embeddings"]
    else:
        print("Arquivo de embeddings não encontrado. Gerando novos embeddings...")

        # Carregar dataset e pré-processar
        print("Carregando e processando dataset...")
        dataset = load_dataset("waitmandot/test", split="train")

        # Processar e converter a nova estrutura do dataset
        documents = []
        for idx, record in enumerate(dataset):
            document = {
                "id": record["metadata"]["id"],
                "title": record["metadata"]["title"],
                "tags": record["metadata"]["tags"],
                "created_at": record["metadata"]["created_at"],
                "content": record["content"]["text"],
                "summary": record["content"]["summary"],
                "context": {
                    "preceding_text": record["context"]["preceding_text"],
                    "following_text": record["context"]["following_text"]
                }
            }
            documents.append(document)

        print(f"{len(documents)} documentos carregados e processados.")

        # Gerar embeddings para cada documento
        print("Gerando embeddings para os documentos...")
        document_embeddings = []
        for doc in documents:
            embedding = get_embedding(doc["content"])
            document_embeddings.append({
                "id": doc["id"],
                "embedding": embedding,
                "payload": doc
            })

        # Salvar documentos e embeddings em arquivo
        with open(embedding_file_path, "w") as f:
            json.dump({"documents": documents, "embeddings": document_embeddings}, f)
        print(f"Documentos e embeddings salvos em '{embedding_file_path}'.")

        return documents, document_embeddings

# Carregar ou gerar os embeddings
documents, document_embeddings = carregar_ou_gerar_embeddings()

# API para consulta
class Consulta(BaseModel):
    pergunta: str

class Resposta(BaseModel):
    resultados: List[dict]

@app.post("/consulta", response_model=Resposta)
async def consultar(dados: Consulta):
    try:
        # Converter embeddings carregados em array NumPy
        embeddings = np.array([doc["embedding"] for doc in document_embeddings])

        # Obter embedding da pergunta
        pergunta_embedding = np.array(get_embedding(dados.pergunta))

        # Calcular similaridades (usando similaridade cosseno)
        similarities = cosine_similarity([pergunta_embedding], embeddings)[0]

        # Obter os 5 documentos mais similares
        top_indices = similarities.argsort()[-5:][::-1]

        # Combinar documentos e notas de similaridade
        resultados = [
            {
                "similarity_score": round(float(similarities[i]), 2),
                "document": documents[i]
            }
            for i in top_indices
        ]

        return {"resultados": resultados}
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
