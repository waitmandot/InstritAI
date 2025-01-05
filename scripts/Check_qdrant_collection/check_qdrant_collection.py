import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# Carregar a chave API do arquivo .env
load_dotenv()
QDRANT_KEY = os.getenv("QDRANT_KEY")

if not QDRANT_KEY:
    raise ValueError("A chave API QDRANT_KEY não foi encontrada no arquivo .env")

# Inicializar o cliente Qdrant
qdrant_client = QdrantClient(
    url="https://704ff291-d513-44da-adfd-46e78acfa20a.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=QDRANT_KEY,
)

# Nome da coleção
COLLECTION_NAME = "teste"

try:
    # Verificar se a coleção existe
    if qdrant_client.get_collection(COLLECTION_NAME):
        print(f"A coleção '{COLLECTION_NAME}' já existe. Nenhuma ação será realizada.")
    else:
        raise Exception("Coleção não encontrada.")  # Forçar o bloco de criação
except Exception:
    # Criar a coleção se não existir
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=100, distance=models.Distance.COSINE),
    )
    print(f"A coleção '{COLLECTION_NAME}' foi criada com sucesso!")
