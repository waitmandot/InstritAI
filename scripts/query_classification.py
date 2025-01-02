import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Configuração da API
API_KEY = os.getenv("API_KEY")  # Certifique-se de configurar sua chave no .env
API_URL = os.getenv("API_URL")  # URL da API do modelo Llama
MODEL = "meta-llama/llama-3.2-3b-instruct:free"

# Inicialize a lista para armazenar o log da sessão
session_log = []

# Verifica se as variáveis necessárias estão configuradas
if not API_KEY or not API_URL:
    raise ValueError("API_KEY ou API_URL não estão configuradas corretamente. Verifique o arquivo .env.")

# Função para enviar uma consulta ao modelo
def ask_model(query):
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

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json().get("choices", [{}])[0].get("text", "").strip()
        # Adicionando a mensagem "É necessário consultar RAG para essa resposta?"
        result = f"É necessário consultar RAG para essa resposta? {result}"

        # Adicionando a interação ao log da sessão
        session_log.append(f"Query: {query}\nResponse: {result}\n" + "-" * 40 + "\n")

        # Salva o log completo da sessão no arquivo
        with open("query_log.txt", "w") as log_file:
            log_file.writelines(session_log)

        return result
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {e}")
        return None


# Loop principal para interação pelo console
if __name__ == "__main__":
    print("AI Console - Type your query or 'exit' to quit.")
    while True:
        user_input = input("Query: ").strip()
        if user_input.lower() in ["exit", "quit", "fechar", "close"]:
            print("Exiting the console. Goodbye!")
            break
        response = ask_model(user_input)
        if response:
            print(f"AI: {response}")