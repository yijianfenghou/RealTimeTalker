import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    PROJECT_NAME = "STT-LLM-TTS App"
    VERSION = "1.0.0"

    # Google Cloud credentials
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    # Ollama settings
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

    # Supported languages
    SUPPORTED_LANGUAGES = {
        'en-US': 'English (US)',
        'zh-CN': 'Chinese (Simplified)',
        'ja-JP': 'Japanese',
        'ko-KR': 'Korean'
    }


settings = Settings()