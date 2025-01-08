import requests

API_URL = "http://127.0.0.1:8000/calcular-similaridades"

def main():
    print("Bem-vindo ao cliente de consulta de embeddings!")
    while True:
        print("\nMenu:")
        print("1. Fazer uma nova consulta")
        print("2. Sair")
        escolha = input("Escolha uma opção: ")

        if escolha == "1":
            consultas = []
            passagens = []

            print("\nDigite suas consultas. Digite 'done' para finalizar:")
            while True:
                consulta = input("Consulta: ")
                if consulta.lower() == "done":
                    break
                consultas.append(consulta)

            print("\nDigite suas passagens. Digite 'done' para finalizar:")
            while True:
                passagem = input("Passagem: ")
                if passagem.lower() == "done":
                    break
                passagens.append(passagem)

            if not consultas or not passagens:
                print("Por favor, forneça pelo menos uma consulta e uma passagem!")
                continue

            # Cria o payload
            payload = {"consultas": consultas, "passagens": passagens}

            try:
                # Faz a requisição para o servidor FastAPI
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()  # Levanta exceção para erros HTTP

                # Processa a resposta
                resultado = response.json()
                print("\nResultados:")
                for i, pontuacoes in enumerate(resultado["pontuacoes"]):
                    print(f"Consulta {i + 1}: {consultas[i]}")
                    for j, pontuacao in enumerate(pontuacoes):
                        print(f"  Passagem {j + 1}: {passagens[j]} - Similaridade: {pontuacao:.2f}")

            except requests.exceptions.RequestException as e:
                print(f"Erro ao se conectar ao servidor: {e}")

        elif escolha == "2":
            print("Saindo...")
            break
        else:
            print("Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()
