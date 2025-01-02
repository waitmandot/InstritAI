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
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

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
def search_query_in_qdrant(search_query, qdrant_client, top_k=3):
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
    # Determina se é necessário consultar o Qdrant
    if query_classification(user_query):
        # Se `query_classification` retornar True, faz a pesquisa no Qdrant
        results = search_query_in_qdrant(user_query, qdrant_client, top_k=3)
        source_knowledge = "\n".join(results)
        augment_prompt = f"""
            You are FixIt, an assistant specialized in industrial machinery. Use the context below to answer the question. If the question is not related to the context, provide a generic response.

            ### Context
            {source_knowledge}

            ### Question
            {user_query}

            ### Response Instructions
            1. Please answer clearly, concisely, objectively and briefly and in Portuguese (Brazil).
            2. If necessary to list information, use bullet points or enumeration.
            3. Whenever possible, justify your response based on the provided context.
            4. Avoid making inferences outside the context.
            5. Be polite, maintain a professional tone, and prioritize safety.
        """
    else:
        # Caso contrário, responde genericamente
        augment_prompt = f"""
            You are FixIt, an assistant specialized in industrial machinery. Answer the question directly.

            ### Question
            {user_query}

            ### Response Instructions
            1. Please answer clearly, concisely, objectively and briefly and in Portuguese (Brazil).
            2. Be polite, maintain a professional tone, and prioritize safety.
        """

    print(augment_prompt)

    # Envia o prompt para o modelo e processa a resposta
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


def query_classification(query):
    prompt = f"""
    You are a technical AI assistant specialized in industrial machinery and maintenance practices. Your task is to determine whether answering a question requires consulting technical documentation, manuals, or detailed records. Your response must be either "y" (for yes) or "n" (for no), without any variation in formatting, spacing, or punctuation.

    ### When to respond "y":
    1. The question explicitly mentions:
       - Manuals, guides, technical documents, or records.
    2. The question requires specific details that are:
       - Machine-specific (e.g., part numbers, capacities, or tolerances).
       - Dependent on manufacturer recommendations or standards.
    3. The requested information impacts:
       - Equipment safety, reliability, or operational efficiency.

    ### When to respond "n":
    1. The question is conversational, generic, or conceptual (e.g., "What is maintenance?").
    2. The answer is widely understood without reference to specific documentation.
    3. The question does not involve technical precision or machine-specific details.

    ### Examples:
    - "What is the difference between corrective and preventive maintenance?" → n
    - "How much oil does a WEG X123 compressor need?" → y
    - "Do I need to read the manual to maintain a CNC machine?" → y
    - "What are the benefits of lubrication?" → n
    - "What type of grease is recommended for high-speed bearings in the WEG ABC123 motor?" → y
    - "Can I use synthetic oil for general machinery lubrication?" → n
    - "According to the manual, what is the correct torque for bolts on a CNC lathe?" → y

    ### Output Rules:
    - Respond ONLY with "y" or "n".
    - Do not include any punctuation, spaces, or symbols in your response.
    - Do not provide explanations, variations, or additional information.

    Question: {query}
    Answer:
    """

    payload = {
        "model": MODEL,
        "prompt": prompt.strip(),
        "max_tokens": 2,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json().get("choices", [{}])[0].get("text", "").strip()

    if result.lower() == "y":
        return True
    else:
        return False



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
