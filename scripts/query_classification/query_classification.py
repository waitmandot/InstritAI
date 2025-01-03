import os
import requests
from dotenv import load_dotenv
import time  # Importa o módulo de tempo
from deep_translator import GoogleTranslator

load_dotenv()

# Configuração da API
API_KEY = os.getenv("API_KEY")  # Certifique-se de configurar sua chave no .env
API_URL = os.getenv("API_URL")  # URL da API do modelo Llama
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
    1. The question explicitly or implicitly requires:
       - Machine-specific details (e.g., part numbers, torque values, tolerances, operational parameters).
       - Manufacturer recommendations, such as maintenance schedules, intervals, or operational guidelines.
       - Detailed procedural instructions (e.g., assembly, disassembly, calibration, alignment).
    2. The requested information has implications for:
       - Safety, reliability, or performance of equipment.
       - Compliance with industry or manufacturer standards.
    3. The answer depends on:
       - Specific technical records, manuals, or manufacturer-provided recommendations.
    
    #### Respond "n" if:
    1. The question is conceptual, generic, or educational (e.g., definitions, comparisons).
    2. The information can be provided using widely known principles without referencing specific documentation.
    3. It does not involve technical precision, safety-critical details, or manufacturer-specific data.
    
    ### Examples:
    - "What is predictive maintenance and how does it work?" → n
    - "What is the recommended operating pressure for an Atlas Copco GA30 compressor?" → y
    - "How do I disassemble and reassemble a centrifugal pump?" → y
    - "Can I use common grease for high-speed bearings?" → n
    - "What is the ideal torque for bolts on a WEG 50 HP motor?" → y
    - "What are the main types of bearing failures?" → n
    - "What hydraulic fluid is recommended for high loads at -20°C?" → y
    - "What is the difference between corrective and preventive maintenance?" → n
    - "How to measure the alignment of a motor with coupled equipment?" → y
    
    ### Rules for Output:
    1. Respond **only** with "y" or "n".
    2. Do not include any additional text, punctuation, or spaces.
    3. Maintain a consistent response format for every query.
    
    ### Decision Process:
    1. **Keyword Identification**: Look for terms indicating machine-specific details, procedural actions, or manufacturer dependencies (e.g., "procedures," "recommended," "specific torque").
    2. **Context Evaluation**: Determine if the question addresses a general concept or requires precise technical information.
    3. **Apply the Guidelines**: Based on the context and keywords, classify the query as requiring RAG (y) or not requiring RAG (n).
    
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
        response.raise_for_status()
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
