import fitz  # PyMuPDF
import json
import re
import os
from typing import List

# Define os diretórios de entrada e saída
INPUT_DIRECTORY = "input_files"
OUTPUT_DIRECTORY = "output_files"
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Remove arquivos antigos do diretório de saída
def limpar_diretorio_saida():
    for old_file_name in os.listdir(OUTPUT_DIRECTORY):
        old_file_path = os.path.join(OUTPUT_DIRECTORY, old_file_name)
        if os.path.isfile(old_file_path):
            os.remove(old_file_path)
            print(f"Arquivo {old_file_name} excluído.")

# Função para dividir texto em chunks respeitando frases
def dividir_em_chunks_avancado(texto: str, limite: int) -> List[str]:
    frases = re.split(r'(?<=[.!?]) +', texto)
    chunks, chunk_atual = [], []
    tamanho_atual = 0

    for frase in frases:
        if tamanho_atual + len(frase) <= limite:
            chunk_atual.append(frase)
            tamanho_atual += len(frase)
        else:
            chunks.append(" ".join(chunk_atual).strip())
            chunk_atual = [frase]
            tamanho_atual = len(frase)

    if chunk_atual:
        chunks.append(" ".join(chunk_atual).strip())

    return chunks

# Função para limpar e formatar o texto para melhor leitura por IA
def limpar_e_formatar_texto(texto_original: str) -> str:
    texto_processado = re.sub(r'\s+', ' ', texto_original)  # Substitui múltiplos espaços por um único
    texto_processado = re.sub(r'[^\w\s.,!?-]', '', texto_processado)  # Remove caracteres especiais
    texto_processado = texto_processado.strip()  # Remove espaços extras
    return texto_processado

# Função principal para processar os arquivos PDF
def processar_pdfs():
    limpar_diretorio_saida()

    LIMITE_CARACTERES = 500  # Define o limite de caracteres por chunk

    for pdf_file_name in os.listdir(INPUT_DIRECTORY):
        if pdf_file_name.endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIRECTORY, pdf_file_name)
            documento = fitz.open(pdf_path)
            dados_extraidos = []

            for numero_pagina in range(len(documento)):
                pagina_atual = documento.load_page(numero_pagina)
                texto_pagina = pagina_atual.get_text("text")

                # Remove marcações de páginas e limpa o texto
                texto_pagina = re.sub(r'Página\s+\d+|Page\s+\d+', '', texto_pagina, flags=re.IGNORECASE)
                texto_limpo = limpar_e_formatar_texto(texto_pagina)

                # Divide o texto em chunks
                chunks_extraidos = dividir_em_chunks_avancado(texto_limpo, LIMITE_CARACTERES)

                for idx, chunk in enumerate(chunks_extraidos):
                    chunk_data = {
                        "chunk-id": f"{numero_pagina}-{idx}",
                        "chunk": chunk,
                        "title": os.path.splitext(pdf_file_name)[0],
                        "page-number": numero_pagina + 1
                    }
                    dados_extraidos.append(chunk_data)

            # Salva os dados extraídos em um arquivo JSON
            output_file_path = os.path.join(OUTPUT_DIRECTORY, f"{os.path.splitext(pdf_file_name)[0]}.json")
            with open(output_file_path, "w", encoding='utf-8') as json_output_file:
                json.dump(dados_extraidos, json_output_file, ensure_ascii=False, indent=4)

            print(f"Extração concluída para {pdf_file_name}. JSON salvo em {output_file_path}.")

if __name__ == "__main__":
    processar_pdfs()
