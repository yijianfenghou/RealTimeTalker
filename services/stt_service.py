from google.cloud import speech
import asyncio
from typing import AsyncGenerator


class STTService:
    def __init__(self):
        self.client = speech.SpeechClient()

    async def transcribe_streaming(self, audio_stream, language_code="en-US") -> AsyncGenerator[str, None]:
        """
        Performs streaming speech recognition on audio content.
        """
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=language_code,
            enable_automatic_punctuation=True,
        )

        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )

        requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                    for chunk in audio_stream)

        responses = self.client.streaming_recognize(streaming_config, requests)

        for response in responses:
            for result in response.results:
                if result.is_final:
                    yield result.alternatives[0].transcript