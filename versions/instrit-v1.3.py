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

# Load environment variables
load_dotenv()

# OpenRouter API Configuration
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

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

    def get_embedding(self, context: str, model: str = "nomic-embed-text") -> Any | None:
        """Get embeddings using the Nomic API."""
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": context}
        )
        if response.status_code == 200:
            return response.json().get("embedding")
        print(f"[ERROR] Failed to get embedding: {response.text}")
        return None

    def load_and_prepare_data(self):
        """Load and prepare dataset."""
        print("[LOG] Loading dataset...")
        dataset = load_dataset("waitmandot/test", split="train")
        print("[LOG] Converting dataset to Pandas DataFrame...")
        data = dataset.to_pandas()
        print("[LOG] Selecting relevant columns...")
        docs = data[['chunk', 'title']]
        print("[LOG] Loading documents in expected format...")
        loader = DataFrameLoader(docs, page_content_column="chunk")
        return loader.load()

    def generate_embeddings(self, documents_list):
        """Generate embeddings sequentially."""
        print("[LOG] Starting embeddings generation...")
        embeddings_list = []
        total_docs = len(documents_list)
        for idx, doc in enumerate(documents_list, 1):
            embedding = self.get_embedding(doc.page_content)
            if embedding:
                embeddings_list.append(embedding)
            else:
                print(f"[ERROR] Failed to generate embedding for document {idx}/{total_docs}.")
            print(f"[LOG] Progress: {idx}/{total_docs} documents processed.")
        return embeddings_list

    def initialize_qdrant(self, docs, embed_list):
        """Initialize and configure Qdrant."""
        print("[LOG] Configuring Qdrant...")
        self.qdrant_client = QdrantClient(":memory:")

        if not self.qdrant_client.collection_exists(collection_name="chatbot"):
            self.qdrant_client.create_collection(
                collection_name="chatbot",
                vectors_config=VectorParams(size=len(embed_list[0]), distance=Distance.COSINE)
            )

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embed_list[i],
                payload={"content": doc.page_content}
            )
            for i, doc in enumerate(docs)
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

    def query_classification(self, query: str):
        prompt = f"""
        You are a technical AI assistant specialized in industrial machinery and maintenance practices. Your task is to determine whether answering a question requires consulting technical documentation, manuals, or detailed records. Your response must be either "y" (for yes) or "n" (for no), without any variation in formatting, spacing, or punctuation.
        
        ### When to respond "y":
        1. The question explicitly mentions:
           - Manuals, guides, technical documents, or records.
        2. The question requires specific details that are:
           - Machine-specific (e.g., part numbers, capacities, or tolerances).
           - Dependent on manufacturer recommendations or standards.
        3. The requested information impacts:
           - Equipment safety, reliability, or operational efficiency.
        
        ### When to respond "n":
        1. The question is conversational, generic, or conceptual (e.g., "What is maintenance?").
        2. The answer is widely understood without reference to specific documentation.
        3. The question does not involve technical precision or machine-specific details.
        
        ### Examples:
        - "What is the difference between corrective and preventive maintenance?" → n
        - "How much oil does a WEG X123 compressor need?" → y
        - "Do I need to read the manual to maintain a CNC machine?" → y
        - "What are the benefits of lubrication?" → n
        - "What type of grease is recommended for high-speed bearings in the WEG ABC123 motor?" → y
        - "Can I use synthetic oil for general machinery lubrication?" → n
        - "According to the manual, what is the correct torque for bolts on a CNC lathe?" → y
        - "According to documents, what are the review deadlines for industrial equipment?" → y
        - "Can you list the most common types of lubrication?" → y
        - "What are the potential parts for lubrication on a lathe?" → y
        - "What is a compressor?" → n
        
        ### Output Rules:
        - Respond ONLY with "y" or "n".
        - Do not include any punctuation, spaces, or symbols in your response.
        - Do not provide explanations, variations, or additional information.
        
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

        response = requests.post(API_URL, headers=headers, json=payload)
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
            You are Instrict, an assistant specialized in industrial machinery. Use the context below to answer the question. If the question is not related to the context, provide a generic response.

            ### Chat History
            {chat_history}

            ### Context
            {source_knowledge}

            ### Current Question
            {query}

            ### Response Instructions
            1. Please answer clearly, concisely, objectively and briefly and in Portuguese (Brazil).
            2. If necessary to list information, use bullet points or enumeration.
            3. Whenever possible, justify your response based on the provided context.
            4. Maintain consistency with previous responses.
            5. Avoid making inferences outside the context.
            6. Be polite, maintain a professional tone, and prioritize safety.
            """
        else:
            print("[CHAT] 💬 Using conversation mode without database search")
            prompt = f"""
            You are Instrict, an assistant specialized in industrial machinery. Answer the question directly.

            ### Chat History
            {chat_history}

            ### Current Question
            {query}

            ### Response Instructions
            1. Please answer clearly, concisely, objectively and briefly and in Portuguese (Brazil).
            2. Maintain consistency with previous responses.
            3. Be polite, maintain a professional tone, and prioritize safety.
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
            "Authorization": f"Bearer {API_KEY}",
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
        self.initialize_qdrant(loaded_documents, generated_embeddings)

        print("[LOG] Chatbot initialized and ready.")

        while True:
            user_input = input("Você: ")

            if user_input.lower() in ["sair", "fechar"]:
                print("Conversa encerrada.")
                break
            elif user_input.lower() == "/json":
                print(json.dumps(self.conversation_history, indent=4))
                continue

            response = self.generate_response(user_input)
            print("Assistente:", response)


if __name__ == "__main__":
    chatbot = EnhancedChatbot()
    chatbot.run()