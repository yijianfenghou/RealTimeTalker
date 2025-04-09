from google.cloud import texttospeech
import base64


class TTSService:
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

    async def synthesize_speech(self, text: str, language_code: str = "en-US") -> bytes:
        """
        Synthesizes speech from the input text.
        """
        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )

        response = self.client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        return response.audio_content