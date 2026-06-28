from abc import ABC, abstractmethod
from typing import List, Dict

class ChatClient(ABC):

    @abstractmethod
    def generate_response(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        """Call the LLM with conversational context and return the text response."""
        pass
