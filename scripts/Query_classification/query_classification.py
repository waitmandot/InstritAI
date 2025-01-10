import os
import requests
from dotenv import load_dotenv
import time  # Importa o módulo de tempo
from deep_translator import GoogleTranslator

load_dotenv()

# Configuração da API
OPENROUTER_KEY = os.getenv("API_KEY")
API_URL = os.getenv("https://openrouter.ai/api/v1/chat/completions")
MODEL = "meta-llama/llama-3.2-3b-instruct:free"

# Verifica se as variáveis necessárias estão configuradas
if not API_KEY or not API_URL:
    raise ValueError("API_KEY ou API_URL não estão configuradas corretamente. Verifique o arquivo .env.")

# Inicialize a lista global para armazenar o log da sessão
session_log = [f"Script inicializado\n{'-' * 40}\n"]

# Caminho do log (relativo ao diretório de execução)
log_dir = os.path.join(os.getcwd(), "log")  # Cria o diretório log no mesmo local do script
log_file_path = os.path.join(log_dir, "interaction_log.txt")

# Verifica se o diretório existe, se não, cria o diretório
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# Função para enviar uma consulta ao modelo
def ask_model(query):
    prompt = f"""
    You are an advanced technical AI assistant specialized in industrial machinery, maintenance practices, and operational standards. Your primary task is to determine whether a question requires consulting technical documentation, manuals, or detailed records (referred to as RAG). Respond with "y" (yes) or "n" (no), strictly following the guidelines below.
    
    ### Guidelines:
    
    #### Respond "y" if:
    1. The question requires any form of technical, detailed, or specific information, including but not limited to:
       - Definitions of machinery components or their functions (e.g., "What are the parts of a lathe?").
       - Maintenance practices or guidelines (e.g., "How to perform preventive maintenance on a milling machine?").
       - Lubricants, fluids, or material specifications (e.g., "What oil should be used for a refrigerator?").
       - Operational, assembly, or disassembly instructions.
       - Any information about a specific machine model, brand, or type (e.g., "Parts of an industrial fan," "How to troubleshoot an XYZ Model 500?").
       - Descriptions or classifications of machines or their functions.
    
    #### Respond "n" if:
    1. The question involves superficial or conversational inputs (e.g., "Hello," "Who are you?").
    2. It reflects a continuation or exploration of a symptom or issue without requiring documentation (e.g., "It is making noise," "The machine stopped working.").
    3. It is explicitly not technical or specific enough to require reference materials.
    
    ### Examples:
    - "What are the parts of a lathe?" → y
    - "How to perform preventive maintenance on a milling machine?" → y
    - "What oil should be used for a refrigerator?" → y
    - "Parts of an industrial fan?" → y
    - "Hello, who are you?" → n
    - "It is making noise." → n
    - "What are the main types of lubrication?" → y
    - "What is a lathe?" → y
    - "The machine stopped working." → n
    - "What are the benefits of using hydraulic systems in heavy machinery?" → y
    
    ### Rules for Output:
    1. Respond **only** with "y" or "n".
    2. Do not include any additional text, punctuation, or spaces.
    3. Maintain a consistent response format for every query.
    
    ### Decision Process:
    1. **Keyword Identification**: Identify terms that indicate a need for technical details, machine parts, or operational information.
    2. **Context Evaluation**: Assess whether the question involves a technical inquiry or a continuation of a previously described issue.
    3. **Apply the Guidelines**: Classify the query based on the provided rules and respond accordingly.
    
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

    # Marca o tempo antes do envio
    request_send_time = time.time()

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        print("Debug: API response JSON:", response.json())
        response.raise_for_status()  # Levanta uma exceção se o status não for 2xx
        result = response.json().get("choices", [{}])[0].get("text", "").strip()

        # Marca o tempo após a resposta
        response_receive_time = time.time()

        # Calculando o tempo de envio e resposta
        time_to_receive = response_receive_time - request_send_time

        # Adicionando a mensagem "É necessário consultar RAG para essa resposta?"
        result = f"É necessário consultar RAG para essa resposta? {result}"

        # Adicionando a interação ao log da sessão com tempos
        session_log.append(f"Usuário: {query}\nIA: {result}\n"
                           f"Tempo de resposta: {time_to_receive:.2f} seconds\n"
                           f"{'-' * 40}\n")

        # Salva o log completo da sessão no arquivo (sobrescreve o arquivo)
        with open(log_file_path, "w") as log_file:
            log_file.writelines(session_log)

        result_output = (f"{result}\n"
                         f"Tempo de resposta: {time_to_receive:.2f} seconds\n"
                         f"{'-' * 40}")

        return result_output
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {e}")
        return None


# Loop principal para interação pelo console
if __name__ == "__main__":
    print (f"Script inicializado\n{'-' * 40}")
    while True:
        # Desabilita a entrada de novas queries enquanto aguarda resposta
        user_input = input("Usuário: ").strip()
        if user_input.lower() in ["exit", "quit", "fechar", "close"]:
            print("Exiting the console. Goodbye!")
            break

        user_input = GoogleTranslator(source='auto', target='en').translate(user_input)

        # Envia a consulta e espera pela resposta da IA
        response = ask_model(user_input)

        # Imprime a resposta assim que for recebida
        if response:
            print(f"IA: {response}")
