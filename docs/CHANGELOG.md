# Changelog

## instrit-v1.1
Release Date: 31/12/2024

The initial version of the Industrial Maintenance AI Assistant, **instrit-v1.1**, introduces the foundation of the project. It leverages a Retrieval-Augmented Generation (RAG) pipeline to assist in industrial maintenance by answering queries based on a specialized dataset. Here are the main features:

### Key Features:
- **Dataset Integration:**
  - Utilizes a custom dataset hosted on Hugging Face, containing industrial maintenance procedures and troubleshooting guides.
  - Loads and processes data using the `datasets` library and converts it into a searchable format.

- **Knowledge Retrieval:**
  - Integrates **Qdrant**, an in-memory vector database, for efficient retrieval of relevant documents.
  - Embeddings generated using the **Nomic embedding API**.
  - Supports top-k search queries to provide context-specific answers.

- **Language Generation:**
  - Utilizes the OpenRouter API to generate human-like responses.
  - Customizable model parameters such as temperature, token limit, and penalties to fine-tune response behavior.

- **Interactive Interface:**
  - Includes a conversational prompt system that dynamically augments user queries with context retrieved from the database.

- **Environment Variables for Flexibility:**
  - API keys, URLs, and model parameters can be configured via a `.env` file.

### Usage Workflow:
1. **Load Dataset:** The system initializes by loading a dataset and processing it into embeddings for searchability.
2. **Setup Qdrant:** Documents and embeddings are indexed in Qdrant for fast retrieval.
3. **Respond to Queries:** User queries are processed, relevant data is retrieved, and a context-aware response is generated using OpenRouter.

### Notes:
- This is the foundational release, setting the groundwork for future enhancements.
- The system is currently optimized for English, with plans to support Portuguese-Brazil in upcoming versions.

### Known Limitations:
- Limited dataset scope may restrict the accuracy of responses.
- Embedding generation relies on a locally hosted Nomic API, requiring additional setup.
- No predictive maintenance or multilingual support yet.

Stay tuned for updates and enhancements in future versions!

