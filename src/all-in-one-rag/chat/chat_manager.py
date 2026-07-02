from typing import List, Dict, Any
from models.chat import ChatClient
from models.rag import AbstractRAGPipeline
from cross_encoder.cross_encoder import CrossEncoder

class ChatManager:

    def __init__(
        self, 
        rag_pipeline: AbstractRAGPipeline, 
        chat_client: ChatClient, 
        system_prompt: str = "You are a helpful assistant. Answer the user's queries using the provided context when available.",
        max_history: int = 10
    ):
        self._rag_pipeline = rag_pipeline
        self._chat_client = chat_client
        self._system_prompt = system_prompt
        self._max_history = max_history
        self._messages: List[Dict[str, str]] = []
        self._cross_encoder = CrossEncoder()

    def clear_history(self) -> None:
        """Reset the conversation memory."""
        self._messages = []

    def get_history(self) -> List[Dict[str, str]]:
        """Return the current conversation messages."""
        return self._messages

    def chat(self, query_text: str) -> str:
        """
        Coordinate retrieval from RAG pipeline, construct prompt, invoke the LLM,
        and manage conversation memory.
        """
        results = self._rag_pipeline.query(query_text, top_k=20)
        
        if results:
            try:
                top_results = self._cross_encoder.get_tok_k_documents(query_text , results)
                print("Cross encoder ranked successfully")
                context_text = "\n".join([f"- {r}" for r in top_results])
            except Exception as e:
                print(f"\n[Warning] Cross-encoder ranking failed: {e}. Falling back to default retriever order.")
                context_text = "\n".join([f"- {r['text']}" for r in results])
        else:
            context_text = "No relevant context found in database."

        self._messages.append({"role": "user", "content": query_text})

        llm_messages = list(self._messages)
        llm_messages[-1] = {
            "role": "user",
            "content": f"Query: {query_text}\n\nContext:\n{context_text}"
        }

        try:
            response = self._chat_client.generate_response(self._system_prompt, llm_messages)
        except Exception as e:
            response = f"[Error generating response from LLM client]: {e}"

        self._messages.append({"role": "assistant", "content": response})

        if len(self._messages) > self._max_history:
            self._messages = self._messages[1:]

        return response
