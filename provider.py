import os
from abc import ABC, abstractmethod
import google.generativeai as genai
import openai
import ollama

class ModelProvider(ABC):
    """Abstract base class for model providers."""

    @classmethod
    @abstractmethod
    def provider_instance(cls, model_name: str) -> bool:
        """Check if this provider can handle the given model name."""
        pass

    @property
    @abstractmethod
    def env_var_name(self) -> str:
        """The name of the environment variable required for the API key."""
        pass

    @abstractmethod
    def generate_content(self, prompt: str) -> str:
        """Generates content based on the prompt."""
        pass


class GeminiProvider(ModelProvider):
    """Provider for Google's Gemini models."""
    
    def __init__(self, config_api_key: str, model_name: str):
        # In GeminiProvider the order is first config_api_key for backward compatibility
        self.api_key = config_api_key or os.getenv(self.env_var_name)
        if not self.api_key:
            raise ValueError(f"Missing API key for {model_name}. Env var = {self.env_var_name}.")

        self.model_name = model_name
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        """For backward compatibility, Gemini is the default provider."""
        return True

    @property
    def env_var_name(self) -> str:
        return "GEMINI_API_KEY"
        
    def generate_content(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class OpenAIProvider(ModelProvider):
    """Provider for OpenAI models."""
    
    def __init__(self, config_api_key: str, model_name: str):
        self.api_key = os.getenv(self.env_var_name, config_api_key)
        if not self.api_key:
            raise ValueError(f"Missing API key for {model_name}. Env var = {self.env_var_name}.")

        self.model_name = model_name
        self.client = openai.OpenAI(api_key=self.api_key)

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        return model_name.startswith("gpt") or model_name.startswith("o1")

    @property
    def env_var_name(self) -> str:
        return "OPENAI_API_KEY"
        
    def generate_content(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


class OllamaProvider(ModelProvider):
    """Provider for Ollama models."""

    def __init__(self, config_api_key: str, model_name: str):
        self.api_key = os.getenv(self.env_var_name, config_api_key)
        self.model_name = model_name.lstrip("ollama/")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.client = ollama.Client(
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            headers=headers
        )

    @classmethod
    def provider_instance(cls, model_name: str) -> bool:
        return model_name.startswith("ollama/")

    @property
    def env_var_name(self) -> str:
        return "OLLAMA_API_KEY"

    def generate_content(self, prompt: str) -> str:
        response = self.client.chat(
            self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content
