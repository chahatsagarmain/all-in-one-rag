# 🚀 All-in-One RAG Pipeline

```text
    ___   __    __        ____        ____  _   __ ______
   /   | / /   / /       /  _/___    / __ \/ | / // ____/
  / /| |/ /   / /________/ // __ \  / / / /  |/ // __/   
 / ___ / /___/ /___/____/ // / / / / /_/ / /|  // /___   
/_/  |_/_____/_____/   /___/_/ /_/  \____/_/ |_//_____/
                  ____  ___   ______
                 / __ \/   | / ____/
                / /_/ / /| |/ / __  
               / _, _/ ___ / /_/ /  
              /_/ |_/_/  |_|\____/
```

Welcome to the **All-in-One RAG Pipeline**, a modular, high-performance Retrieval-Augmented Generation framework built from scratch to be completely plug-and-play. Swap chunkers, embeddings, storage layers, and chat providers dynamically directly from the command line!

---

## ✨ Features

- 📂 **Multi-Source Ingester**: Seamlessly parse and extract text & metadata from `.pdf`, `.md`, and `.txt` files.
- ✂️ **Advanced Chunking**: Supports token-based fixed-size chunking and similarity-based semantic chunking powered by **Chonkie**.
- 🧬 **Flexible Embeddings**:
  - **Local**: Hugging Face Sentence Transformers (`all-MiniLM-L6-v2`) running entirely locally.
  - **OpenAI**: Cloud-based embeddings using OpenAI's API.
  - **Static**: Super-fast word-embedding projections using local `Model2Vec` (Potion).
- 🗄️ **Persistent Vector Store**: Custom batch uploading and similarity searching in **Weaviate** running via Docker.
- 💬 **Conversational Chat**:
  - Chat memory preserving rolling conversation history.
  - **OpenAI** client integration (GPT models).
  - **Google Gemini** client integration (Gemini 1.5 Flash) via the modern `google-genai` SDK.
  - **Mock Client** fallback for offline execution.

---

## 🛠️ How to Run Locally

Follow these steps to configure and run the interactive CLI on your system:

### 1. Prerequisites
Ensure you have the following installed:
- [Docker & Docker Desktop](https://www.docker.com/)
- [Astral uv](https://github.com/astral-sh/uv) (recommended fast Python package manager) or standard Python 3.11+

### 2. Setup Dependencies & Environment
Clone the repository and install packages:
```bash
# Install dependencies into virtual environment
uv sync
```

Create a `.env` file in the root directory and add your API keys:
```env
# Optional, required if using OpenAI components
OPENAI_API_KEY=your_openai_api_key_here

# Optional, required if using Gemini components
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Spin up Weaviate Database
Launch Weaviate using Docker Compose:
```bash
docker compose up -d
```
Verify Weaviate is running at `http://localhost:8080`.

### 4. Start the RAG CLI
Run the application:
```bash
uv run src/all-in-one-rag/main.py
```

---

## 🎮 Walkthrough of Choices at Each Stage

When you launch the CLI, you will be guided through 5 configuration stages:

### Stage 1: Document Selection
Enter the path to your source document. Press Enter to use the default `README.md` file:
```text
📂 Enter path to PDF / Markdown / TXT file [./README.md]: 
```

### Stage 2: Chunking Strategy
Choose how to divide your document into pieces:
```text
Select the Chunking Model:
  1. Fixed Size Chunking (Token-based)
  2. Semantic Chunking (Split by topic flow)
👉 Choice (1 or 2) [1]:
```

### Stage 3: Embedding Method
Select how to convert text chunks into numerical vectors:
```text
Select the Embedding Method:
  1. Sentence Transformers (Local MiniLM model - 384d)
  2. OpenAI Embeddings (Cloud API - 1536d)
  3. Static Vectors (Ultra-fast local Model2Vec - 512d)
👉 Choice (1, 2, or 3) [1]:
```

### Stage 4: Chat LLM Provider
Select the generator LLM for the chat assistant:
```text
Select the Chat Model / LLM Provider:
  1. OpenAI (gpt-4o-mini)
  2. Google Gemini (gemini-1.5-flash)
  3. Offline / Mock Mode
👉 Choice (1, 2, or 3) [1]:
```

### Stage 5: Conversational Loop
Start chatting! The CLI will fetch relevant contexts from Weaviate, inject them into the LLM prompt, and print responses:
```text
💬 Let's start chatting! (Type 'exit' or 'quit' to stop)

You > What is all-in-one-rag?
RAG > All-in-One RAG is a modular, plug-and-play framework designed to let you dynamically swap...
```