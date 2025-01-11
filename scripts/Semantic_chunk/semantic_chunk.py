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


def summarize(message):
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


def format_to_json(message):
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
        "max_tokens": 0,
        "chat_memory": 8,
        "temperature": 0.2,
        "top_p": 1.0,
        "top_k": 50,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.2,
        "min_p": 0.0,
        "top_a": 0.0,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a professional assistant designed to structure textual input into a well-organized JSON format. "
                    "Your task is to transform the given document page into separate JSON objects for each section. Each JSON object must represent a title and its corresponding paragraph as a single document.\n\n"
                    "### Input Instructions:\n"
                    "1. The input will be a full page of text, formatted with section titles followed by paragraphs.\n"
                    "2. Each section (title + paragraph) should be transformed into its own JSON object.\n\n"
                    "### Output JSON Format:\n"
                    "The output must be in English and follow this exact structure:\n"
                    "{\n"
                    '    "metadata": {\n'
                    '        "id": "",  // Leave this field blank for later completion.\n'
                    '        "source": {\n'
                    '            "file_name": "",  // Leave this field blank for later completion.\n'
                    '            "page_number": ""  // Leave this field blank for later completion.\n'
                    "        },\n"
                    '        "title": "Title of the section",\n'
                    '        "tags": ["tag1", "tag2", "tag3"],  // Generate 3-5 tags based on the content.\n'
                    '        "created_at": ""  // Leave this field blank.\n'
                    "    },\n"
                    '    "content": {\n'
                    '        "text": "Full text of the paragraph goes here.",\n'
                    '        "summary": "Short summary of the paragraph in one or two sentences."\n'
                    "    },\n"
                    '    "context": {\n'
                    '        "preceding_text": "Text preceding this section in the input, if any.",\n'
                    '        "following_text": "Text following this section in the input, if any."\n'
                    "    }\n"
                    "}\n\n"
                    "### Requirements:\n"
                    "1. Leave the fields id, file_name, page_number, and created_at blank in the output.\n"
                    "2. Remove the fields author, last_updated, and related_chunks.\n"
                    "3. Generate meaningful tags based on the content.\n"
                    "4. Include preceding_text and following_text to preserve document flow.\n"
                    "5. Use the section titles as the title field and ensure proper capitalization.\n\n"
                    "### Example Input:\n"
                    "=== Page 5 Summary ===\n\n"
                    "**Friction Concept**\n\n"
                    "Friction is a resistance opposing motion, measured as a force called frictional force. It exists in all types of motion among solids, liquids, or gases. In solids, friction manifests as the resistance encountered when moving one body over another. Friction greatly influences human life, acting either for or against motion, depending on the context.\n\n"
                    "**Types of Friction**\n\n"
                    "Friction can be classified into different types, with the least friction found in gases, followed by fluids, and finally solids. Fluid friction is always less than solid friction. Friction among solids is the highest and can cause problems such as heat generation, energy loss, noise, and wear.\n\n"
                    "**Importance of Lubrication**\n\n"
                    "Lubrication involves introducing a fluid substance between two surfaces to avoid solid-to-solid contact, creating fluid friction instead. This is crucial because it prevents overheating, energy loss, noise, and wear. Additionally, lubrication reduces friction, improving the efficiency and durability of machines and equipment.\n\n"
                    "### Example Output:\n"
                    "[\n"
                    '    {\n'
                    '        "metadata": {\n'
                    '            "id": "",\n'
                    '            "source": {\n'
                    '                "file_name": "",\n'
                    '                "page_number": ""\n'
                    '            },\n'
                    '            "title": "Friction Concept",\n'
                    '            "tags": ["friction", "physics", "motion", "resistance"],\n'
                    '            "created_at": ""\n'
                    '        },\n'
                    '        "content": {\n'
                    '            "text": "Friction is a resistance opposing motion, measured as a force called frictional force. It exists in all types of motion among solids, liquids, or gases. In solids, friction manifests as the resistance encountered when moving one body over another. Friction greatly influences human life, acting either for or against motion, depending on the context.",\n'
                    '            "summary": "Explains the concept of friction, its definition, and its effects on motion."\n'
                    '        },\n'
                    '        "context": {\n'
                    '            "preceding_text": "",\n'
                    '            "following_text": "Friction can be classified into different types..."\n'
                    '        }\n'
                    '    },\n'
                    '    {\n'
                    '        "metadata": {\n'
                    '            "id": "",\n'
                    '            "source": {\n'
                    '                "file_name": "",\n'
                    '                "page_number": ""\n'
                    '            },\n'
                    '            "title": "Types of Friction",\n'
                    '            "tags": ["friction", "types", "solids", "fluids", "gases"],\n'
                    '            "created_at": ""\n'
                    '        },\n'
                    '        "content": {\n'
                    '            "text": "Friction can be classified into different types, with the least friction found in gases, followed by fluids, and finally solids. Fluid friction is always less than solid friction. Friction among solids is the highest and can cause problems such as heat generation, energy loss, noise, and wear.",\n'
                    '            "summary": "Classifies friction into types and discusses its effects."\n'
                    '        },\n'
                    '        "context": {\n'
                    '            "preceding_text": "Friction is a resistance opposing motion...",\n'
                    '            "following_text": "Lubrication involves introducing a fluid substance..."\n'
                    '        }\n'
                    '    },\n'
                    '    {\n'
                    '        "metadata": {\n'
                    '            "id": "",\n'
                    '            "source": {\n'
                    '                "file_name": "",\n'
                    '                "page_number": ""\n'
                    '            },\n'
                    '            "title": "Importance of Lubrication",\n'
                    '            "tags": ["lubrication", "machines", "efficiency", "friction"],\n'
                    '            "created_at": ""\n'
                    '        },\n'
                    '        "content": {\n'
                    '            "text": "Lubrication involves introducing a fluid substance between two surfaces to avoid solid-to-solid contact, creating fluid friction instead. This is crucial because it prevents overheating, energy loss, noise, and wear. Additionally, lubrication reduces friction, improving the efficiency and durability of machines and equipment.",\n'
                    '            "summary": "Describes the role of lubrication in reducing friction and improving efficiency."\n'
                    '        },\n'
                    '        "context": {\n'
                    '            "preceding_text": "Friction can be classified into different types...",\n'
                    '            "following_text": ""\n'
                    '        }\n'
                    '    }\n'
                    "]\n\n"
                    "### Your task:\n"
                    "Process the provided input according to these instructions and output the result as JSON, matching the format and example provided. Ensure strict adherence to the structure."
                )
            },
            {
                "role": "user",
                "content": message
            }
        ]
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
                    # Processamento por página
                    for page_number, page in enumerate(pdf.pages, start=1):

                        # Extrai o texto bruto sem imagens do PDF
                        text = page.extract_text() or ""

                        # Traduz o texto bruto para o inglês
                        translated_text = GoogleTranslator(source='auto', target='en').translate(text)

                        # Envia o texto traduzido para ser limpo
                        cleaned_text = clean_text(translated_text)
                        print(f"Processando página {page_number} do arquivo {file_name}...")

                        # Envia o texto limpo para a IA interpretar
                        summarized_text = summarize(cleaned_text)

                        # Envia o texto interpretado para a IA formatar para json
                        formatted_text = format_to_json(summarized_text)

                        consolidated_file.write(f"{formatted_text}\n\n")

                print(f"Arquivo processado: {file_name}")

    print(f"Resumo consolidado salvo em: {consolidated_output}")

if __name__ == "__main__":
    extract_text_from_pdfs()
