import os
import requests
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# Carregar variáveis de ambiente
load_dotenv()
NOMIC_API_URL = os.getenv("NOMIC_API_URL", "http://localhost:11434/api/embeddings")
NOMIC_MODEL = os.getenv("NOMIC_MODEL", "nomic-embed-text")


# Função para obter embedding
def get_embedding(text: str):
    response = requests.post(NOMIC_API_URL, json={"model": NOMIC_MODEL, "prompt": text})
    if response.status_code == 200:
        return response.json().get("embedding")
    else:
        print(f"Erro ao gerar embedding: {response.text}")
        return None


# Loop para aguardar e processar perguntas
while True:
    # Solicitar entrada do usuário
    question = input("Digite uma pergunta para gerar o embedding (ou 'sair' para encerrar): ")

    # Se o usuário digitar 'sair', o loop será interrompido
    if question.lower() == "sair":
        print("Saindo...")
        break

    question = GoogleTranslator(source='auto', target='en').translate(question)

    # Obter embedding da pergunta
    embedding = get_embedding(question)

    if embedding:
        print("Embedding gerado com sucesso!")
        print(embedding)

        search_query = {
            "vector": embedding,
            "limit": 3,
            "with_payload": True
        }
        print("Código JSON para busca:")
        print(search_query)
