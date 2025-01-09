import os
import pdfplumber
import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Carregar modelo de IA local com logs detalhados
def carregar_modelo_local(modelo_nome="EleutherAI/gpt-neo-125M"):
    print("[INFO] Iniciando o carregamento do modelo...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(modelo_nome)
        print("[INFO] Tokenizer carregado com sucesso.")

        modelo = AutoModelForCausalLM.from_pretrained(modelo_nome)
        print("[INFO] Modelo de IA carregado com sucesso.")

        ia_pipeline = pipeline("text-generation", model=modelo, tokenizer=tokenizer, device=-1)
        print("[INFO] Pipeline configurado com sucesso.")
        return ia_pipeline, tokenizer
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao carregar o modelo: {e}")
        raise

# Dividir texto em chunks respeitando o limite do modelo e tokens reais
def dividir_em_chunks(texto, tokenizer, max_tokens=1024):
    print("[INFO] Dividindo texto em chunks...")
    tokens = tokenizer.encode(texto, truncation=False)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))

    print(f"[INFO] Texto dividido em {len(chunks)} chunks.")
    return chunks

# Processar texto com IA local
def processar_texto_ia(ia_pipeline, texto, max_new_tokens=300):
    print("[INFO] Processando texto com IA...")
    prompt = (
        "Organize o seguinte texto em tópicos claros, com títulos, resuma informações desnecessárias e prepare para um dataset de RAG: \n\n"
        f"{texto}\n\n"
        "Responda com um JSON estruturado assim: "
        '{"tópicos": [{"título": "Título do Tópico", "conteúdo": "Resumo ou conteúdo estruturado"}]}'
    )
    try:
        resposta = ia_pipeline(prompt, max_new_tokens=max_new_tokens, num_return_sequences=1)
        print("[INFO] Texto processado com sucesso pela IA.")
        return resposta[0]["generated_text"]
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao processar o texto com a IA: {e}")
        return None

# Função para extrair texto do PDF
def extrair_texto_pdf(caminho_pdf):
    print(f"[INFO] Extraindo texto do PDF: {caminho_pdf}")
    try:
        with pdfplumber.open(caminho_pdf) as pdf:
            texto_total = ""
            for pagina in pdf.pages:
                texto_total += pagina.extract_text() + "\n"
        print("[INFO] Extração de texto concluída com sucesso.")
        return texto_total
    except Exception as e:
        print(f"[ERRO] Ocorreu um erro ao extrair texto do PDF: {e}")
        return ""

# Estruturar dataset em JSON
def criar_dataset_json(respostas_ia, caminho_saida):
    print(f"[INFO] Criando dataset JSON em: {caminho_saida}")
    try:
        dataset = {"tópicos": []}
        for resposta in respostas_ia:
            if resposta:
                try:
                    chunk_dataset = json.loads(resposta)
                    dataset["tópicos"].extend(chunk_dataset.get("tópicos", []))
                except json.JSONDecodeError:
                    print("[ERRO] Resposta inválida ignorada. Não foi possível processar o JSON.")

        with open(caminho_saida, "w", encoding="utf-8") as arquivo:
            json.dump(dataset, arquivo, indent=4, ensure_ascii=False)
        print(f"[INFO] Dataset salvo com sucesso em: {caminho_saida}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar o dataset JSON: {e}")
        raise

# Processar todos os arquivos PDF no diretório input_files
def processar_arquivos_diretorio(diretorio_input, diretorio_output, modelo_nome="EleutherAI/gpt-neo-125M"):
    print("[INFO] Iniciando processamento dos arquivos no diretório...")
    ia_pipeline, tokenizer = carregar_modelo_local(modelo_nome)

    arquivos = [f for f in os.listdir(diretorio_input) if f.endswith('.pdf')]
    if not arquivos:
        print("[INFO] Nenhum arquivo PDF encontrado no diretório.")
        return

    for arquivo in arquivos:
        print(f"[INFO] Processando arquivo: {arquivo}...")
        caminho_pdf = os.path.join(diretorio_input, arquivo)
        nome_arquivo = os.path.splitext(arquivo)[0]
        caminho_saida = os.path.join(diretorio_output, f"{nome_arquivo}.json")

        texto_pdf = extrair_texto_pdf(caminho_pdf)
        if not texto_pdf.strip():
            print(f"[ERRO] Arquivo {arquivo} vazio ou inválido. Ignorando.")
            continue

        chunks = dividir_em_chunks(texto_pdf, tokenizer, max_tokens=512)  # Ajustar para lidar com modelos menores
        respostas_ia = [processar_texto_ia(ia_pipeline, chunk) for chunk in chunks]

        criar_dataset_json(respostas_ia, caminho_saida)

# Diretórios de entrada e saída
diretorio_input = "./input_files"
diretorio_output = "./output_files"

os.makedirs(diretorio_output, exist_ok=True)

# Executar
if __name__ == "__main__":
    processar_arquivos_diretorio(diretorio_input, diretorio_output)
