import requests
import json
from datasets import load_dataset
from langchain_community.document_loaders import DataFrameLoader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do .env
load_dotenv()

# Configuração da API do OpenRouter
OPENROUTER_KEY = os.getenv("API_KEY")
API_URL = os.getenv("https://openrouter.ai/api/v1/chat/completions")

# Parâmetros do modelo
MODEL = os.getenv("MODEL", "meta-llama/llama-3.2-3b-instruct:free")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 600))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
TOP_P = float(os.getenv("TOP_P", 1))
TOP_K = int(os.getenv("TOP_K", 0))
FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", 0.5))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", 0.5))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.10))
MIN_P = float(os.getenv("MIN_P", 0))
TOP_A = float(os.getenv("TOP_A", 0))

# Inicialização da conversa
with open("../system_prompt.json", "r") as file:
    system_prompt = json.load(file)
conversation_history = [system_prompt]

# Função para obter embeddings usando a API Nomic
def get_embedding(context, model="nomic-embed-text"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": context}
    )
    if response.status_code == 200:
        return response.json().get("embedding")
    print(f"[ERRO] Falha ao obter embedding: {response.text}")
    return None

# Função para carregar o conjunto de dados e preparar documentos
def load_and_prepare_data():
    print("[LOG] Carregando conjunto de dados...")
    dataset = load_dataset("waitmandot/test", split="train")
    print("[LOG] Convertendo conjunto de dados para DataFrame Pandas...")
    data = dataset.to_pandas()
    print("[LOG] Selecionando colunas relevantes...")
    docs = data[['chunk', 'title']]
    print("[LOG] Carregando documentos no formato esperado...")
    loader = DataFrameLoader(docs, page_content_column="chunk")
    return loader.load()

# Função para gerar embeddings de forma sequencial
def generate_embeddings(documents_list):
    print("[LOG] Iniciando geração de embeddings...")
    embeddings_list = []
    total_docs = len(documents_list)
    print(f"[LOG] Progresso: 0/{total_docs} documentos processados.")
    for idx, doc in enumerate(documents_list, 1):
        embedding = get_embedding(doc.page_content)
        if embedding:
            embeddings_list.append(embedding)
        else:
            print(f"[ERRO] Falha ao gerar embedding para o documento {idx}/{total_docs}.")
        print(f"[LOG] Progresso: {idx}/{total_docs} documentos processados.")
    return embeddings_list

# Função para configurar e adicionar documentos ao Qdrant
def add_documents_to_qdrant(docs, embed_list):
    print("[LOG] Configurando Qdrant...")
    qdrant_client = QdrantClient(":memory:")  # Usa Qdrant em memória
    print("[LOG] Criando coleção no Qdrant...")

    if not qdrant_client.collection_exists(collection_name="chatbot"):
        qdrant_client.create_collection(
            collection_name="chatbot",
            vectors_config=VectorParams(size=len(embed_list[0]), distance=Distance.COSINE)
        )

    print("[LOG] Adicionando documentos ao Qdrant...")
    points = [
        PointStruct(
            id=str(uuid.uuid4()),  # Gera UUID único para cada ponto
            vector=embed_list[i],
            payload={"content": doc.page_content}
        )
        for i, doc in enumerate(docs)
    ]

    qdrant_client.upsert(collection_name="chatbot", points=points)

    print("[LOG] Documentos adicionados com sucesso.")
    return qdrant_client

# Função para realizar busca no Qdrant
def search_query_in_qdrant(search_query, qdrant_client, top_k=2):
    print(f"[LOG] Gerando embedding da pergunta...")
    embedding = get_embedding(search_query)
    if not embedding:
        print("[ERRO] Falha ao gerar embedding para a consulta.")
        return []
    print("[LOG] Buscando no Qdrant...")
    search_result = qdrant_client.search(
        collection_name="chatbot",
        query_vector=embedding,
        limit=top_k,
    )
    print("[LOG] Retornando resposta...")
    return [result.payload["content"] for result in search_result]

# Função para lidar com prompts personalizados
def custom_prompt(user_query, qdrant_client):
    results = search_query_in_qdrant(user_query, qdrant_client, top_k=3)
    source_knowledge = "\n".join(results)
    augment_prompt = f"""Use o contexto abaixo para responder à pergunta e se não houver relação responda normalmente.

    Contexto:
    {source_knowledge}

    Pergunta: {user_query}"""

    print(augment_prompt)

    # Envia para o modelo e processa a resposta
    conversation_history.append({"role": "user", "content": augment_prompt})

    payload = {
        "model": MODEL,
        "messages": conversation_history,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "frequency_penalty": FREQUENCY_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "repetition_penalty": REPETITION_PENALTY,
        "min_p": MIN_P,
        "top_a": TOP_A,
        "transforms": ["middle-out"]
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        model_response = response.json()["choices"][0]["message"]["content"]
        print("Assistente:", model_response)
    else:
        print(f"[ERRO API] {response.status_code} - {response.text}")

if __name__ == "__main__":
    print("[LOG] Iniciando pipeline...")

    # Carregar e preparar os dados
    loaded_documents = load_and_prepare_data()

    # Gerar embeddings
    generated_embeddings = generate_embeddings(loaded_documents)
    if not generated_embeddings:
        print("[ERRO] Nenhum embedding foi gerado. Encerrando programa.")
        exit(1)

    # Adicionar documentos ao Qdrant
    qdrant_client_instance = add_documents_to_qdrant(loaded_documents, generated_embeddings)

    while True:
        user_input = input("Você: ")

        if user_input.lower() in ["sair", "fechar"]:
            print("Conversa encerrada.")
            break
        elif user_input.lower() == "/json":
            print(json.dumps(conversation_history, indent=4))
            continue

        custom_prompt(user_input, qdrant_client_instance)
