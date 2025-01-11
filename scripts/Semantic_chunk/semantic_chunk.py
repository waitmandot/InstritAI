import os
import re
import uuid
from datetime import datetime
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
                    "[\n"
                        "{\n"
                    '       "metadata": {\n'
                    '           "id": "",  // Leave this field blank for later completion.\n'
                    '           "source": {\n'
                    '               "file_name": "",  // Leave this field blank for later completion.\n'
                    '               "page_number": ""  // Leave this field blank for later completion.\n'
                    "           },\n"
                    '           "title": "Title of the section",\n'
                    '           "tags": ["tag1", "tag2", "tag3"],  // Generate 3 tags (words) based on the content.\n'
                    '           "created_at": ""  // Leave this field blank.\n'
                    "       },\n"
                    '       "content": {\n'
                    '           "text": "Full text of the paragraph goes here.",\n'
                    '           "summary": "Short summary of the paragraph in one or two sentences."\n'
                    "       },\n"
                    '       "context": {\n'
                    '           "preceding_text": "Text preceding this section in the input, if any.",\n'
                    '           "following_text": "Text following this section in the input, if any."\n'
                    "       }\n"
                    "   }\n"
                    "]\n\n"
                    "### Requirements:\n"
                    "1. Leave the fields id, file_name, page_number, and created_at blank in the output.\n"
                    "2. Remove the fields author, last_updated, and related_chunks.\n"
                    "3. Generate meaningful tags based on the content.\n"
                    "4. Include preceding_text and following_text to preserve document flow.\n"
                    "5. Use the section titles as the title field and ensure proper capitalization.\n\n"
                    "### Example Input:\n"
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
                    '            "tags": ["physics", "motion", "resistance"],\n'
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
                    '            "tags": ["types", "fluids", "gases"],\n'
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
                    '            "tags": ["machines", "efficiency", "friction"],\n'
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
    consolidated_output = os.path.join(output_dir, "consolidated_summary.json")
    consolidated_data = []

    # Variável para contar as chamadas à IA
    ia_requests_count = 0

    # Início do cronômetro
    start_time = datetime.now()

    # Itera pelos arquivos na pasta de entrada
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".pdf"):
            input_path = os.path.join(input_dir, file_name)

            # Extrai o texto do PDF
            with pdfplumber.open(input_path) as pdf:
                total_pages = len(pdf.pages)
                print(f"Processando arquivo: {file_name} ({total_pages} páginas)")

                # Processamento por página
                for page_number, page in enumerate(pdf.pages, start=1):
                    try:
                        # Data e hora do processamento por página
                        page_processing_time = datetime.now().isoformat()

                        # Extrai o texto bruto sem imagens do PDF
                        text = page.extract_text() or ""
                        if not text.strip():
                            print(f"A página {page_number} do arquivo {file_name} está vazia. Ignorando.")
                            continue

                        # Traduz o texto bruto para o inglês
                        translated_text = GoogleTranslator(source='auto', target='en').translate(text)

                        # Verifica se o texto traduzido está vazio
                        if not translated_text.strip():
                            print(f"Texto traduzido vazio na página {page_number} do arquivo {file_name}. Ignorando.")
                            continue

                        # Envia o texto traduzido para ser limpo
                        cleaned_text = clean_text(translated_text)
                        if not cleaned_text.strip():
                            print(f"Texto limpo na página {page_number} do arquivo {file_name} está vazio. Ignorando.")
                            continue

                        print(f"Processando página {page_number}/{total_pages} do arquivo {file_name}...")

                        # Envia o texto limpo para a IA interpretar
                        summarized_text = summarize(cleaned_text)
                        ia_requests_count += 1  # Incrementa a contagem para summarize
                        if not summarized_text.strip():
                            print(f"Resumo gerado vazio na página {page_number} do arquivo {file_name}. Ignorando.")
                            continue

                        # Repetição começa na formatação para JSON
                        page_processed = False
                        while not page_processed:
                            try:
                                # Envia o texto interpretado para a IA formatar para JSON
                                formatted_text = format_to_json(summarized_text)
                                ia_requests_count += 1  # Incrementa a contagem para format_to_json

                                # Remove caracteres especiais indesejados e mantém apenas os válidos para JSON
                                formatted_text = re.sub(r'[^a-zA-Z0-9,\[\]{}:\-\"\s_]', '', formatted_text)

                                # Usando expressão regular pra capturar tudo entre colchetes []
                                match = re.search(r'\[(.*)]$', formatted_text, re.DOTALL)
                                if not match:
                                    print(f"JSON válido não encontrado na página {page_number} do arquivo {file_name}.")
                                    print(f"Texto retornado pela IA: {formatted_text}")
                                    raise ValueError("JSON inválido gerado pela IA.")

                                json_text = match.group(1)  # Pega o conteúdo do JSON sem os colchetes

                                # Converte o texto JSON em um dicionário Python
                                json_dict = json.loads(f"[{json_text}]")  # Certifica que o texto seja interpretado como JSON válido

                                # Adiciona os campos adicionais necessários
                                for element in json_dict:
                                    element["metadata"]["id"] = str(uuid.uuid4())  # Gera um ID único
                                    element["metadata"]["source"]["file_name"] = file_name  # Adiciona o nome do arquivo
                                    element["metadata"]["source"]["page_number"] = page_number  # Adiciona o número da página
                                    element["metadata"]["created_at"] = page_processing_time  # Adiciona a data e hora por página

                                consolidated_data.extend(json_dict)  # Adiciona cada objeto à lista consolidada
                                page_processed = True  # Marca a página como processada com sucesso
                            except Exception as e:
                                print(f"Erro ao processar JSON na página {page_number} do arquivo {file_name}: {e}")
                                print("Tentando novamente...")
                                print(f"Processando página {page_number}/{total_pages} do arquivo {file_name}...")
                                # A execução continuará repetindo até que seja bem-sucedida

                    except Exception as e:
                        print(f"Erro geral ao processar a página {page_number} do arquivo {file_name}: {e}")

    # Salva o JSON consolidado
    with open(consolidated_output, "w", encoding="utf-8") as consolidated_file:
        json.dump(consolidated_data, consolidated_file, indent=4, ensure_ascii=False)

    # Tempo total de execução
    end_time = datetime.now()
    total_time = end_time - start_time

    # Formatação do tempo total em horas, minutos ou segundos
    if total_time.total_seconds() < 60:
        formatted_time = f"{total_time.total_seconds():.2f} segundo" if total_time.total_seconds() == 1 else f"{total_time.total_seconds():.2f} segundos"
    elif total_time.total_seconds() < 3600:
        minutes, seconds = divmod(total_time.total_seconds(), 60)
        formatted_time = (
                             f"{int(minutes)} minuto" if minutes == 1 else f"{int(minutes)} minutos"
                         ) + " e " + (
                             f"{int(seconds)} segundo" if seconds == 1 else f"{int(seconds)} segundos"
                         )
    else:
        hours, remainder = divmod(total_time.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        formatted_time = (
                             f"{int(hours)} hora" if hours == 1 else f"{int(hours)} horas"
                         ) + ", " + (
                             f"{int(minutes)} minuto" if minutes == 1 else f"{int(minutes)} minutos"
                         ) + " e " + (
                             f"{int(seconds)} segundo" if seconds == 1 else f"{int(seconds)} segundos"
                         )

    # Exibe o número total de requisições feitas à IA e o tempo total de execução
    print(f"Número total de requisições feitas à IA: {ia_requests_count}")
    print(f"Tempo total de execução: {formatted_time}")
    print(f"Resumo consolidado salvo em: {consolidated_output}")


if __name__ == "__main__":
    extract_text_from_pdfs()
