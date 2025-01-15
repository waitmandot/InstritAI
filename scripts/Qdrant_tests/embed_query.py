import os
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Carregar a chave API do arquivo .env
load_dotenv()
QDRANT_KEY = os.getenv("QDRANT_KEY")

if not QDRANT_KEY:
    raise ValueError("A chave API QDRANT_KEY não foi encontrada no arquivo .env")

# Configurações
NOMIC_API_URL = "http://localhost:11434/api/embeddings"
NOMIC_MODEL = "nomic-embed-text"
COLLECTION_NAME = "chatbot"

# Inicializar cliente Qdrant
qdrant_client = QdrantClient(
    url="https://704ff291-d513-44da-adfd-46e78acfa20a.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=QDRANT_KEY,
)

# Função para obter embedding da pergunta
def get_embedding(text: str, model: str = NOMIC_MODEL):
    response = requests.post(
        NOMIC_API_URL,
        json={"model": model, "prompt": text}
    )
    if response.status_code == 200:
        return response.json().get("embedding")
    else:
        raise ValueError(f"Erro ao gerar embedding: {response.status_code} - {response.text}")

# Obter o embedding da pergunta
question = "What does a lubrication system typically consist of?"
question_embedding = get_embedding(question)

# Realizar a busca no Qdrant
search_results = qdrant_client.search(
    collection_name=COLLECTION_NAME,
    query_vector=question_embedding,
    limit=5
)

# Exibir resultados
for result in search_results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score}")
    print(f"Payload: {result.payload}")
    print("-" * 50)
