import os
import pdfplumber
import re
import requests
import json
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração da API
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def clean_text(text):
    # Remove linhas vazias, múltiplos espaços, sequências de hifens, underlines e espaços pontilhados
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove linhas vazias
    text = re.sub(r' {2,}', ' ', text)  # Substitui múltiplos espaços por um único
    text = re.sub(r'[_]{4,}', '', text)  # Remove sequências de underlines soltos com mais de 4 caracteres
    text = re.sub(r'-{4,}', '', text)  # Remove sequências de hifens soltos com mais de 4 caracteres
    text = re.sub(r'\.{2,}', '', text)  # Remove sequências de pontos
    text = re.sub(r'\s+\.{2,}\s+', ' ', text)  # Remove linhas com espaços pontilhados
    return text

def make_request(message):
    """
    Faz uma requisição para a IA para processar o texto.

    Args:
        message (str): Conteúdo a ser enviado ao modelo.

    Returns:
        str: Resposta do modelo.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
    }

    payload = {
        "model": "meta-llama/llama-3.1-8b-instruct:free",
        "top_p": 0.9,
        "temperature": 0.3,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.2,
        "top_k": 50,
        "messages": [
            {
                "role": "system",
                "content": (
                    "The response should be in English."
                    "You are an AI specialized in processing and summarizing technical content. "
                    "Your task is to extract the key concepts from the provided content, organizing them into a clear and coherent summary. "
                    "You should focus on the technical aspects, ignoring irrelevant or disconnected information (such as names, places, or institutions), and ensure that the summary maintains a logical flow. "
                    "The output must consist of multiple paragraphs, each with a focused subtitle. "
                    "Each paragraph should concentrate on one primary concept, clearly explaining it while maintaining clarity and conciseness. "
                    "Ensure that the information is logically sequenced and free from extraneous details, avoiding unnecessary separations, markdown characters, line breaks, or lists."
                    "Your goal is to create a structured and cohesive summary that captures the essence of the technical content, while discarding any non-relevant references."
                )
            },
            {"role": "user", "content": message},
        ],
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response content")
    else:
        return f"Error: {response.status_code}, {response.text}"

def extract_text_from_pdfs():
    # Diretórios de entrada e saída
    input_dir = "input_files"
    output_dir = "output_files"

    # Cria os diretórios se não existirem
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Arquivo final para consolidar os resumos
    consolidated_output = os.path.join(output_dir, "consolidated_summary.md")
    with open(consolidated_output, "w", encoding="utf-8") as consolidated_file:

        # Itera pelos arquivos na pasta de entrada
        for file_name in os.listdir(input_dir):
            if file_name.lower().endswith(".pdf"):
                input_path = os.path.join(input_dir, file_name)

                # Extrai o texto do PDF
                with pdfplumber.open(input_path) as pdf:
                    for page_number, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text() or ""
                        text = GoogleTranslator(source='auto', target='en').translate(text)
                        cleaned_text = clean_text(text)
                        print(f"Processando página {page_number} do arquivo {file_name}...")

                        # Envia o texto limpo para a IA
                        summarized_text = make_request(cleaned_text)
                        consolidated_file.write(f"=== Page Summary {page_number} ({file_name}) ===\n\n")
                        consolidated_file.write(f"{summarized_text}\n\n")

                print(f"Arquivo processado: {file_name}")

    print(f"Resumo consolidado salvo em: {consolidated_output}")

if __name__ == "__main__":
    extract_text_from_pdfs()
