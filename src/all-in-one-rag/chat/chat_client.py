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
        import time
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
        
        max_retries = 5
        backoff_delay = 2.0
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config
                )
                return response.text or ""
            except Exception as e:
                err_str = str(e)
                if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str:
                    if attempt < max_retries - 1:
                        time.sleep(backoff_delay)
                        backoff_delay *= 2
                        continue
                raise e
        return ""


class OllamaChatClient(ChatClient):

    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://127.0.0.1:11434"):
        self._model_name = model_name
        self._base_url = base_url.rstrip("/").removesuffix("/v1")

    def generate_response(self, system_prompt: str, messages: List[Dict[str, str]]) -> str:
        import urllib.request
        import json
        
        url = f"{self._base_url}/api/chat"
        
        payload_messages = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            role = "user" if msg["role"] == "user" else "assistant"
            payload_messages.append({"role": role, "content": msg["content"]})
            
        data = {
            "model": self._model_name,
            "messages": payload_messages,
            "stream": False
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data.get("message", {}).get("content", "")
        except Exception as e:
            return (
                f"[Error connecting to local Ollama server at {self._base_url}]: {e}\n\n"
                f"💡 Troubleshooting Tips:\n"
                f"1. Make sure Ollama is installed and running (`ollama serve` or desktop app running).\n"
                f"2. Pull the model first: `ollama pull {self._model_name}`\n"
                f"3. If using Windows/macOS, try using 'http://127.0.0.1:11434' instead of 'http://localhost:11434' (IPv6 resolution of 'localhost' to '::1' may fail if Ollama is only listening on IPv4)."
            )


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
