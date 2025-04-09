from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import asyncio
from services.stt_service import STTService
from services.llm_service import LLMService
from services.tts_service import TTSService
from config import settings
import io

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# 静态文件和模板配置
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# 服务实例化
stt_service = STTService()
llm_service = LLMService(settings.OLLAMA_API_URL, settings.OLLAMA_MODEL)
tts_service = TTSService()


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "supported_languages": settings.SUPPORTED_LANGUAGES}
    )


@app.websocket("/ws/speech")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 接收音频数据
            audio_data = await websocket.receive_bytes()

            # 语音转文字
            async for transcript in stt_service.transcribe_streaming(
                    io.BytesIO(audio_data), language_code="en-US"
            ):
                # 发送转写文本到客户端
                await websocket.send_text(f"Transcript: {transcript}")

                # 使用LLM生成响应
                llm_response = await llm_service.generate_response(transcript)
                await websocket.send_text(f"LLM Response: {llm_response}")

                # 文字转语音
                audio_content = await tts_service.synthesize_speech(
                    llm_response, language_code="en-US"
                )
                await websocket.send_bytes(audio_content)

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await websocket.close()


@app.post("/synthesize")
async def synthesize_text(text: str, language_code: str = "en-US"):
    audio_content = await tts_service.synthesize_speech(text, language_code)
    return StreamingResponse(io.BytesIO(audio_content), media_type="audio/mp3")