import os
import requests
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from datasets import load_dataset

# Carregar variáveis de ambiente
load_dotenv()
QDRANT_KEY = os.getenv("QDRANT_KEY")
NOMIC_API_URL = "http://localhost:11434/api/embeddings"
NOMIC_MODEL = "nomic-embed-text"

# Configurações da coleção
COLLECTION_NAME = "chatbot"
VECTOR_SIZE = 768  # Ajustar conforme o modelo utilizado

# Inicializar cliente Qdrant
print("[LOG] Inicializando cliente Qdrant...")
qdrant_client = QdrantClient(
    url="https://704ff291-d513-44da-adfd-46e78acfa20a.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=QDRANT_KEY,
)

# Verificar ou recriar a coleção
print(f"[LOG] Verificando a existência da coleção '{COLLECTION_NAME}'...")
existing_collections = qdrant_client.get_collections().collections
collection_exists = any(col.name == COLLECTION_NAME for col in existing_collections)

if collection_exists:
    # Verificar se a dimensão está correta
    collection_info = qdrant_client.get_collection(COLLECTION_NAME)
    current_vector_size = collection_info.config.vectors.size
    if current_vector_size != VECTOR_SIZE:
        print(f"[LOG] Dimensão incorreta na coleção '{COLLECTION_NAME}'. Apagando...")
        qdrant_client.delete_collection(COLLECTION_NAME)
        print(f"[LOG] Coleção '{COLLECTION_NAME}' apagada.")

if not collection_exists or current_vector_size != VECTOR_SIZE:
    print(f"[LOG] Criando a coleção '{COLLECTION_NAME}' com dimensão {VECTOR_SIZE}...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[LOG] Coleção '{COLLECTION_NAME}' criada com sucesso.")
else:
    print(f"[LOG] A coleção '{COLLECTION_NAME}' já existe com a dimensão correta.")

# Carregar dataset
print("[LOG] Carregando dataset...")
dataset = load_dataset("waitmandot/test", split="train")

# Processar 10 documentos para fins de teste
print("[LOG] Processando documentos...")
documents = []
for idx, record in enumerate(dataset):
    if idx >= 10:
        break
    documents.append({
        "id": record["metadata"]["id"],
        "title": record["metadata"]["title"],
        "tags": record["metadata"]["tags"],
        "created_at": record["metadata"]["created_at"],
        "content": record["content"]["text"],
        "summary": record["content"]["summary"],
        "preceding_text": record["context"]["preceding_text"],
        "following_text": record["context"]["following_text"]
    })

print(f"[LOG] {len(documents)} documentos processados para teste.")

# Função para obter embeddings
def get_embedding(text: str, model: str = NOMIC_MODEL):
    try:
        response = requests.post(
            NOMIC_API_URL,
            json={"model": model, "prompt": text}
        )
        if response.status_code == 200:
            return response.json().get("embedding")
        else:
            print(f"[ERRO] Resposta inválida da API: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[ERRO] Falha ao obter embedding: {e}")
        return None

# Gerar embeddings e preparar pontos para o Qdrant
print("[LOG] Gerando embeddings...")
points = []
for idx, doc in enumerate(documents):
    embedding = get_embedding(doc["content"])
    if embedding:
        points.append(
            PointStruct(
                id=doc["id"],
                vector=embedding,
                payload={
                    "title": doc["title"],
                    "tags": doc["tags"],
                    "created_at": doc["created_at"],
                    "summary": doc["summary"],
                    "preceding_text": doc["preceding_text"],
                    "following_text": doc["following_text"]
                },
            )
        )
        print(f"[LOG] Documento {idx + 1}/{len(documents)} processado.")
    else:
        print(f"[ERRO] Falha ao gerar embedding para o documento {idx + 1}/{len(documents)}.")

# Subir pontos para o Qdrant
if points:
    print(f"[LOG] Inserindo {len(points)} pontos na coleção '{COLLECTION_NAME}'...")
    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("[LOG] Pontos inseridos com sucesso!")
else:
    print("[ERRO] Nenhum ponto foi gerado. Saindo.")
