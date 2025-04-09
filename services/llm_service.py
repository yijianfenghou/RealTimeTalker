import aiohttp
import json
from typing import Dict, Any


class LLMService:
    def __init__(self, api_url: str, model: str):
        self.api_url = api_url
        self.model = model

    async def generate_response(self, prompt: str) -> str:
        """
        Generate response using Ollama API
        """
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }

            async with session.post(f"{self.api_url}/api/generate",
                                    json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('response', '')
                else:
                    raise Exception(f"Error from Ollama API: {response.status}")