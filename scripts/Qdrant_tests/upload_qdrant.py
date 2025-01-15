import os
import json
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Carregar a chave API do arquivo .env
load_dotenv()
QDRANT_KEY = os.getenv("QDRANT_KEY")

if not QDRANT_KEY:
    raise ValueError("A chave API QDRANT_KEY não foi encontrada no arquivo .env")

# Configurações
COLLECTION_NAME = "chatbot"

# Inicializar cliente Qdrant
qdrant_client = QdrantClient(
    url="https://704ff291-d513-44da-adfd-46e78acfa20a.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=QDRANT_KEY,
)

# Verificar ou criar a coleção
if COLLECTION_NAME not in [col.name for col in qdrant_client.get_collections().collections]:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": 768, "distance": "Cosine"},  # Ajuste o tamanho do vetor
    )

# Carregar arquivo JSON com embeddings
with open("documents_embeddings.json", "r") as file:
    data = json.load(file)

# Preparar pontos para envio
points = [
    PointStruct(
        id=doc["id"],
        vector=embedding["embedding"],
        payload={
            "title": doc["title"],
            "tags": doc["tags"],
            "created_at": doc["created_at"],
            "content": doc["content"],
            "summary": doc["summary"],
            "context": doc["context"],
        },
    )
    for doc, embedding in zip(data["documents"], data["embeddings"])
]

# Inserir os pontos no Qdrant
qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

print("[LOG] Dados enviados ao Qdrant com sucesso!")
