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