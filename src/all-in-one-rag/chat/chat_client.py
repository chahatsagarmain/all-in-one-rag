import os
from typing import List, Dict
from models.chat import ChatClient
from openai import OpenAI
from google import genai
from google.genai import types

class OpenAIChatClient(ChatClient):

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self._model_name = model_name

    def generate_response(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        
        payload = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            payload.append({"role": msg["role"], "content": msg["content"]})
            
        response = client.chat.completions.create(
            model=self._model_name,
            messages=payload
        )
        return response.choices[0].message.content or ""


class GeminiChatClient(ChatClient):

    def __init__(self, model_name: str = "gemini-3.1-flash-lite"):
        self._model_name = model_name
        self._client = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def generate_response(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        client = self._get_client()
        
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )
            
        config = types.GenerateContentConfig(
            system_instruction=system_prompt
        )
        
        response = client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config
        )
        return response.text or ""


class MockChatClient(ChatClient):

    def generate_response(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        last_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        # Extract context snippet if present in the prompt
        context_preview = "No context retrieved."
        if "Context:" in last_user_msg:
            try:
                parts = last_user_msg.split("Context:")
                if len(parts) > 1:
                    context_preview = parts[1].strip()[:200] + "..."
            except Exception:
                pass
                
        return (
            f"[Mock LLM Response]\n"
            f"  - Received Query: '{last_user_msg.split('Context:')[0].strip()}'\n"
            f"  - System Instruction length: {len(system_prompt)} chars\n"
            f"  - Context Preview: {context_preview}\n"
            f"  - Conversation History turns: {len(messages) - 1}"
        )
