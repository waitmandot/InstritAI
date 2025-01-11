import os
import requests
import json
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Obtém a chave de API do arquivo .env
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

def make_request(message):
    """
    Faz uma requisição para o modelo meta-llama/llama-3-8b-instruct:free.

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
        "top_p": 0.9,  # Ajuste para maior foco nas palavras mais prováveis
        "temperature": 0.3,  # Leve aumento na temperatura para maior fluidez
        "frequency_penalty": 0.2,  # Penalização para evitar repetições excessivas
        "presence_penalty": 0.2,  # Penalização leve para maior variedade no vocabulário
        "repetition_penalty": 1.2,  # Leve aumento para controlar mais as repetições
        "top_k": 50,  # Amostragem com as 50 palavras mais prováveis
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

if __name__ == "__main__":
    # Conteúdo da mensagem
    message = """Espírito Santo

Lubrication
Friction
Friction is a generic term for resistance that opposes movement. This resistance is measured by a force denoting frictional force. Discover friction in any type of movement between solids, liquids or gases. In the case of movement between solids, friction can be defined as a resistance that manifests itself when one body moves over another.
Friction has a great influence on human life, either in favor or against. In the first case, for example, making it possible to simply walk. The second concern is in the closest and everything has been done to minimize this force. The least friction that exists is that of gases, followed by that of fluids and, finally, that of solids. Since fluid friction is always less than solid friction, lubrication consists of the interposition of a fluid substance between two surfaces, thus avoiding solid-to-solid contact and producing fluid friction. It is very important to avoid solid-to-solid contact, as this causes the parts to heat up, energy loss due to the parts sticking together, noise and wear. SENAI Espírito Santo Regional Department 5
"""

    # Chama a função e imprime o resultado
    response = make_request(message)
    print(response)
