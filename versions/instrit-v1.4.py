import requests
import json
import uuid
from dotenv import load_dotenv
import os
from datasets import load_dataset
from langchain_community.document_loaders import DataFrameLoader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage
from typing import List, Any
from deep_translator import GoogleTranslator

# Load environment variables
load_dotenv()

# OpenRouter API Configuration
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Qdrant API Configuration
QDRANT_KEY = os.getenv("QDRANT_KEY")

# Model parameters
MODEL = os.getenv("MODEL", "meta-llama/llama-3.2-3b-instruct:free")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 600))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))
TOP_P = float(os.getenv("TOP_P", 1))
TOP_K = int(os.getenv("TOP_K", 0))
FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", 0.5))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", 0.5))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.10))
MIN_P = float(os.getenv("MIN_P", 0))
TOP_A = float(os.getenv("TOP_A", 0))

# Memory Configuration
MEMORY_KEY = "chat_history"
MAX_WINDOW_SIZE = 5  # Number of conversations to remember


class EnhancedChatbot:
    def __init__(self):
        # Initialize system prompt
        with open("../system_prompt.json", "r") as file:
            self.system_prompt = json.load(file)

        # Initialize conversation memory
        self.memory = ConversationBufferWindowMemory(
            memory_key=MEMORY_KEY,
            k=MAX_WINDOW_SIZE,
            return_messages=True
        )

        # Initialize conversation history with system prompt
        self.conversation_history = [self.system_prompt]

        # Initialize Qdrant client
        self.qdrant_client = None

    @staticmethod
    def get_embedding(context: str, model: str = "nomic-embed-text") -> Any | None:
        """Get embeddings using the Nomic API."""
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": context}
        )
        if response.status_code == 200:
            return response.json().get("embedding")
        print(f"[ERROR] Failed to get embedding: {response.text}")
        return None

    @staticmethod
    def load_and_prepare_data():
        """Load and prepare dataset."""
        print("[LOG] Loading dataset...")
        dataset = load_dataset("waitmandot/test", split="train")

        # Parse the new JSON structure
        print("[LOG] Converting dataset to structured format...")
        documents = []
        for idx, record in enumerate(dataset):
            if idx >= 10:  # Process only the first 10 records
                break
            document = {
                "id": record["metadata"]["id"],
                "title": record["metadata"]["title"],
                "tags": record["metadata"]["tags"],
                "created_at": record["metadata"]["created_at"],
                "content": record["content"]["text"],
                "summary": record["content"]["summary"],
                "context": {
                    "preceding_text": record["context"]["preceding_text"],
                    "following_text": record["context"]["following_text"]
                }
            }
            documents.append(document)
        print(f"[LOG] Loaded {len(documents)} documents.")
        return documents

    def generate_embeddings(self, documents_list):
        """Generate embeddings sequentially."""
        print("[LOG] Starting embeddings generation...")
        embeddings_list = []
        total_docs = len(documents_list)
        for idx, doc in enumerate(documents_list, 1):
            embedding = self.get_embedding(doc["content"])
            if embedding:
                embeddings_list.append({
                    "id": doc["id"],
                    "vector": embedding,
                    "payload": doc
                })
            else:
                print(f"[ERROR] Failed to generate embedding for document {idx}/{total_docs}.")
            print(f"[LOG] Progress: {idx}/{total_docs} documents processed.")
        return embeddings_list

    def initialize_qdrant(self, embed_list):
        """Initialize and configure Qdrant."""
        print("[LOG] Configuring Qdrant...")
        self.qdrant_client = QdrantClient(":memory:")

        if not self.qdrant_client.collection_exists(collection_name="chatbot"):
            self.qdrant_client.create_collection(
                collection_name="chatbot",
                vectors_config=VectorParams(size=len(embed_list[0]["vector"]), distance=Distance.COSINE)
            )

        points = [
            PointStruct(
                id=doc["id"],
                vector=doc["vector"],
                payload=doc["payload"]
            )
            for doc in embed_list
        ]

        self.qdrant_client.upsert(collection_name="chatbot", points=points)
        print("[LOG] Documents successfully added to Qdrant.")

    def search_qdrant(self, query: str, top_k: int = 3) -> List[str]:
        """Search for relevant documents in Qdrant."""
        print("[LOG] Generating query embedding...")
        embedding = self.get_embedding(query)
        if not embedding:
            return []

        results = self.qdrant_client.search(
            collection_name="chatbot",
            query_vector=embedding,
            limit=top_k,
        )
        return [result.payload["content"] for result in results]

    @staticmethod
    def query_classification(query: str):
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
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        print("Debug: API response JSON:", response.json())
        response.raise_for_status()
        result = response.json().get("choices", [{}])[0].get("text", "").strip()

        if result.lower() == "y":
            return True
        else:
            return False

    def format_chat_history(self) -> str:
        """Format chat history for context."""
        messages = self.memory.chat_memory.messages
        formatted_history = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_history += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                formatted_history += f"Assistant: {msg.content}\n"
        return formatted_history

    def generate_response(self, query: str) -> str:
        """Generate response using the model."""
        needs_context = self.query_classification(query)
        chat_history = self.format_chat_history()

        if needs_context:
            print("[DATABASE] 🔍 Searching knowledge base for relevant information...")
            results = self.search_qdrant(query, top_k=3)
            if results:
                print("[DATABASE] ✅ Found relevant information in knowledge base")
                print(f"[DATABASE] 📚 Number of relevant documents found: {len(results)}")
            else:
                print("[DATABASE] ❌ No relevant information found in knowledge base")

            source_knowledge = "\n".join(results)

            prompt = f"""
            You are Instrit, an assistant specialized in industrial machinery. Use the documents below to answer the question. If the question is not related to the content of the documents, provide a generic response.

            ### Chat History
            {chat_history}

            ### Context
            {source_knowledge}

            ### Current Question
            {query}

            ### Response Instructions
            1. Provide a **clear, concise, and technically sound answer** in Portuguese (Brazil).  
            2. Use natural, conversational language, avoiding overly formal or mechanical expressions. Aim for the tone of a knowledgeable technician helping a colleague.  
            3. Keep the response brief and to the point, but include enough detail to be practically useful.  
            4. If additional context or elaboration is needed, provide it only in response to follow-up questions.  
            5. If the information is not in the provided documents, politely suggest checking a manual or consulting a specialist.  
            6. Maintain a professional, approachable, and safety-focused tone throughout the response.  
            """
        else:
            print("[CHAT] 💬 Using conversation mode without database search")
            prompt = f"""
            You are Instrit, an assistant specialized in industrial machinery. Answer the question directly, based on your general knowledge.

            ### Chat History
            {chat_history}

            ### Current Question
            {query}

            ### Response Instructions
            1. Answer **clearly, concisely, objectively, and briefly** in Portuguese (Brazil). Avoid over-explaining unless explicitly requested.
            2. Use simple, conversational language, focusing on practical and actionable information.
            3. If the question requires elaboration, wait for follow-up questions before providing more details.
            4. Maintain consistency with previous responses to avoid conflicting information.
            5. Be polite, maintain a professional tone, and prioritize safety. Keep the tone friendly and approachable.
            """

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": prompt})

        # Prepare API request
        payload = {
            "model": MODEL,
            "messages": self.conversation_history,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "frequency_penalty": FREQUENCY_PENALTY,
            "presence_penalty": PRESENCE_PENALTY,
            "repetition_penalty": REPETITION_PENALTY,
            "min_p": MIN_P,
            "top_a": TOP_A,
            "transforms": ["middle-out"]
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            response_content = response.json()["choices"][0]["message"]["content"]

            # Update memory
            self.memory.save_context(
                {"input": query},
                {"output": response_content}
            )

            # Update conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_content
            })

            return response_content
        else:
            error_msg = f"[API ERROR] {response.status_code} - {response.text}"
            print(error_msg)
            return error_msg

    def run(self):
        """Run the chatbot interaction loop."""
        print("[LOG] Starting pipeline...")

        # Load and prepare initial data
        loaded_documents = self.load_and_prepare_data()
        generated_embeddings = self.generate_embeddings(loaded_documents)

        if not generated_embeddings:
            print("[ERROR] No embeddings generated. Exiting program.")
            return

        # Initialize Qdrant
        self.initialize_qdrant(generated_embeddings)

        print("[LOG] Chatbot initialized and ready.")

        while True:
            user_input = input("Você: ")

            if user_input.lower() in ["sair", "fechar", "close", "exit"]:
                print("Conversa encerrada.")
                break
            elif user_input.lower() == "/json":
                print(json.dumps(self.conversation_history, indent=4))
                continue

            user_input = GoogleTranslator(source='auto', target='en').translate(user_input)

            response = self.generate_response(user_input)
            print("Assistente:", response)


if __name__ == "__main__":
    chatbot = EnhancedChatbot()
    chatbot.run()