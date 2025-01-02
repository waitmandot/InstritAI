# Industrial Maintenance AI Assistant

Welcome to my project, **Industrial Maintenance AI Assistant**! This repository represents my journey as an amateur developer learning Artificial Intelligence (AI) while building an assistant designed to aid industrial machine maintenance. This project emphasizes safety, performance, and competitiveness in the industrial sector.

## Table of Contents
- [About the Project](#about-the-project)
- [Importance of AI in Industrial Maintenance](#importance-of-ai-in-industrial-maintenance)
- [Applications](#applications)
- [How It Works](#how-it-works)
  - [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
  - [Dataset](#dataset)
  - [APIs and Tools](#apis-and-tools)
- [Technologies Used](#technologies-used)
- [File Mapping](#file-mapping)
- [Getting Started](#getting-started)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

## About the Project
This project focuses on creating an AI-powered assistant to enhance industrial maintenance operations. The assistant leverages a **simple Retrieval-Augmented Generation (RAG)** approach to provide relevant, contextual answers using a dataset tailored for industrial maintenance tasks.

The primary goals are:
1. **Enhance Safety**: Reduce human error by providing accurate guidance during maintenance operations.
2. **Improve Performance**: Enable faster troubleshooting and diagnostics, ensuring less downtime.
3. **Boost Competitiveness**: Equip industries with cutting-edge technology to maintain operational efficiency and innovation.

This project is being developed under the repository [InstritAI](https://github.com/waitmandot/InstritAI.git), maintained by Cauê Waitman (GitHub: `waitmandot`).

## Importance of AI in Industrial Maintenance
Industrial maintenance is critical to ensuring the longevity and efficiency of machines. AI solutions bring significant advantages:

- **Safety:** AI can prevent accidents by identifying potential issues early and offering precise instructions for handling complex machinery.
- **Cost Efficiency:** Reduce downtime and repair costs by diagnosing problems faster and suggesting preventive measures.
- **Scalability:** AI systems can manage large datasets, enabling predictive maintenance for multiple machines simultaneously.
- **Data-Driven Insights:** Improve decision-making through historical and real-time data analysis.

## Applications
This AI assistant can be applied in:
- Manufacturing plants
- Oil and gas refineries
- Aerospace engineering
- Food processing industries
- Automotive assembly lines

## How It Works
### RAG (Retrieval-Augmented Generation)
The assistant uses a RAG approach to combine:
- **Knowledge Retrieval:** Extract relevant documents or data chunks from a dataset hosted on Hugging Face.
- **Generative AI:** Use OpenRouter APIs to generate context-aware responses.

### Dataset
I created a custom dataset hosted on **Hugging Face** with information about industrial maintenance procedures, troubleshooting guides, and best practices. This dataset ensures the assistant is specialized for its intended use. The focus is on expanding the dataset to support Portuguese-Brazil use cases in the near future.

### APIs and Tools
The assistant integrates the following APIs and tools:
- **OpenRouter APIs** for language generation.
- **Nomic's embedding API** for semantic search.
- **Qdrant** as the vector database for efficient knowledge retrieval.

## Technologies Used
- **Programming Language:** Python (chosen for its flexibility and ecosystem for AI/ML development)
- **Libraries:**
  - `langchain_community`
  - `qdrant-client`
  - `datasets`
- **Database:** Qdrant (in-memory)
- **API Integration:** OpenRouter
- **Environment Management:** dotenv
- **Custom Dataset:** Hugging Face

## File Mapping

### docs/

- **README.md**: Refactored and detailed project documentation.
- **LICENSE**: Licensing file for the project.
- **CHANGELOG.md**: Contains version histories and updates.

### versions/

- **instrit-v1.1.py**: Stable implementation of the initial project version.

### Root Directory

- **.gitignore**: Specifies untracked files for Git.
- **requirements.txt**: Lists required Python dependencies.
- **system_prompt.json**: Contains the initial system prompt for the assistant.

## Getting Started
To set up and run the project on your local machine, follow these steps:

### Prerequisites
1. Python 3.9+
2. Installed dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Environment variables configured in a `.env` file:
   ```env
    API_KEY=<your_openrouter_api_key>
    API_URL=https://openrouter.ai/api/v1/chat/completions
    MODEL=meta-llama/llama-3.2-3b-instruct:free
    MAX_TOKENS=0
    TEMPERATURE=1
    TOP_P=1
    TOP_K=0
    FREQUENCY_PENALTY=0.5
    PRESENCE_PENALTY=0.5
    REPETITION_PENALTY=1.10
    MIN_P=0
    TOP_A=0
   ```

### Running the Project
You can choose between different versions of the execution script:

#### Running `instrit-v1.1.py` (stable version):
1. Clone the repository:
   ```bash
   git clone https://github.com/waitmandot/InstritAI.git
   ```
2. Navigate to the project directory:
   ```bash
   cd InstritAI
   ```
3. Run the script:
   ```bash
   python instrit-v1.1.py
   ```

#### Running `instrit-v1.2.py` (improved version):
1. Make sure the project setup steps are completed.
2. Run the script:
   ```bash
   python instrit-v1.2.py
   ```

## Future Work
- **Expand Dataset:** Add more comprehensive and diverse data, focusing on Portuguese-Brazil use cases.
- **Deploy as a Web App:** Create a user-friendly interface for industries to access the assistant.
- **Integrate Predictive Maintenance:** Use machine learning models to forecast machine failures.
- **Multilingual Support:** Enable the assistant to provide guidance in multiple languages.
- **Scalability and Accessibility:** Ensure the project remains free and accessible, with potential scalability for commercial applications.

## Acknowledgments
- **Hugging Face** for hosting the dataset.
- **OpenRouter** for enabling powerful language generation.
- **Nomic AI** for the embedding API.
- **Qdrant** for efficient vector database management.

---

Thank you for exploring this project! As an amateur developer, this is an exciting opportunity to learn and contribute to the field of AI while addressing real-world industrial challenges. If you have any suggestions or want to collaborate, feel free to reach out!

