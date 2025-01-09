import requests
from deep_translator import GoogleTranslator

API_URL = "http://127.0.0.1:8000/consulta"

def main():
    print("Bem-vindo ao cliente de consulta ao dataset!")
    while True:
        print("\nMenu:")
        print("1. Fazer uma nova consulta")
        print("2. Sair")
        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            pergunta = input("\nDigite sua pergunta: ")

            pergunta = GoogleTranslator(source='auto', target='en').translate(pergunta)

            # Cria o payload
            payload = {"pergunta": pergunta}

            try:
                # Faz a requisição para o servidor FastAPI
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # Levanta exceção para erros HTTP

                # Processa a resposta
                resultado = response.json()
                print("\nParágrafos mais relevantes:")
                for i, chunk in enumerate(resultado["resultados"]):
                    print(f"{i + 1}. {chunk}")

            except requests.exceptions.RequestException as e:
                print(f"Erro ao se conectar ao servidor: {e}")

        elif escolha == "2":
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
