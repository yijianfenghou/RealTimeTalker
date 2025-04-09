import vosk
import sounddevice as sd
import numpy as np
import queue
import json
import threading
import time
import os
import sys
import torch
import ollama # Ollama Python client

# Attempt to set HF_ENDPOINT, useful if downloading models initially
# Note: If using source='local' and models exist, this might not be strictly needed for loading
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- Configuration ---
# Vosk
VOSK_MODEL_PATH = "./models/vosk-model-cn-0.22" # Make sure this path is correct
VOSK_SAMPLE_RATE = 16000
DEVICE_ID = None # Default audio input, find yours using `python -m sounddevice` if needed
BLOCK_SIZE = 4000 # Audio buffer size in frames

# ChatTTS
CHATTS_SAMPLE_RATE = 24000 # ChatTTS default output sample rate
# Determine device for ChatTTS (prioritize GPU)
CHATTS_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHATTS_MODEL_SOURCE = 'local' # Use 'local' if models are in asset dir, 'huggingface' otherwise
CHATTS_LOCAL_PATH = './asset' # Define where ChatTTS should look for local models

# Ollama
OLLAMA_MODEL = "deepseek-v2:latest" # Ensure this model is pulled in Ollama: `ollama pull deepseek-v2:latest`
OLLAMA_HOST = "http://localhost:11433" # Default Ollama API endpoint

# Conversation Settings
MAX_HISTORY_TURNS = 5 # Number of user/assistant pairs to keep in history
SYSTEM_PROMPT = "你是一个友好、乐于助人的中文AI助手。"
EXIT_PHRASES = ["再见", "退出", "拜拜", "结束对话"] # Phrases to trigger exit

# --- Global Variables ---
q = queue.Queue() # Queue for audio data from callback to Vosk listener
vosk_recognizer = None
chat_tts = None # ChatTTS instance placeholder
ollama_client = None # Ollama client instance placeholder
recording_event = threading.Event() # Controls when the audio callback puts data into the queue
conversation_history = [] # Stores conversation context
audio_stream = None # Sounddevice input stream

# --- STT (Vosk) Functions ---
def audio_callback(indata, frames, time, status):
    """This function is called by sounddevice for each audio block."""
    if status:
        print(status, file=sys.stderr)
    # Only queue audio data if the recording event is set (i.e., during active listening)
    if recording_event.is_set():
        q.put(bytes(indata))

def listen_vosk():
    """Listens for speech using Vosk and returns the recognized text."""
    global vosk_recognizer, q
    if not vosk_recognizer:
        print("错误: Vosk 识别器未初始化。")
        return None

    print("\n系统: 请开始说话...")
    recording_event.set() # Signal the callback to start queueing audio data
    # Clear the queue of any stale data before starting
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            continue

    recognized_text = ""
    try:
        while True: # Keep listening until a final result is detected or timeout occurs
            try:
                # Wait for audio data with a timeout
                # The timeout helps detect the end of speech when silence follows
                data = q.get(timeout=1.5)
            except queue.Empty:
                # If the queue is empty after the timeout, it might be the end of speech.
                # Force Vosk to process any remaining buffered data.
                if vosk_recognizer.AcceptWaveform(b''): # Sending empty bytes signals end of stream segment
                   final_result_json = vosk_recognizer.FinalResult()
                   final_result = json.loads(final_result_json)
                   if final_result.get("text"):
                       recognized_text = final_result["text"].strip() # Strip leading/trailing whitespace
                       if recognized_text: # Check if not empty after stripping
                            break # Exit loop on valid final result
                # If still no final result after timeout, continue listening loop briefly
                # print("Timeout, listening again briefly...") # Debugging line
                continue # Continue waiting for more audio

            # Process the received audio data
            if vosk_recognizer.AcceptWaveform(data):
                # A final result was determined based on the processed data
                result_json = vosk_recognizer.FinalResult()
                result = json.loads(result_json)
                if result.get("text"):
                    recognized_text = result["text"].strip()
                    if recognized_text:
                        # print(f"Vosk Final: {recognized_text}") # Debugging line
                        break # Exit loop on valid final result
            # else:
                 # Optional: Display partial results for feedback during speech
                 # partial_result = json.loads(vosk_recognizer.PartialResult())
                 # if partial_result.get("partial"): print(f"Vosk Partial: {partial_result['partial']}", end='\r')

    except KeyboardInterrupt:
        print("\n系统: 收到中断信号。")
        return "EXIT_SIGNAL" # Use a specific signal for graceful exit
    except Exception as e:
        print(f"Vosk 识别错误: {e}")
        return None
    finally:
        recording_event.clear() # Signal the callback to stop queueing audio
        # print("") # Newline after potential partial results
        vosk_recognizer.Reset() # Reset Vosk state for the next listening turn
        # Clear the queue completely after stopping recording
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    if recognized_text:
        print(f"你: {recognized_text}")
        return recognized_text
    else:
        print("系统: 抱歉，没有识别到有效语音。")
        return None

# --- LLM (Ollama) Function ---
def get_llm_response(user_input):
    """Sends conversation history to Ollama and gets a response."""
    global ollama_client, conversation_history

    if not ollama_client:
        print("错误: Ollama 客户端未初始化。")
        return "LLM客户端错误" # Return specific error message

    # Add user input to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Trim history if it exceeds max length (keeping the system prompt)
    if len(conversation_history) > (MAX_HISTORY_TURNS * 2 + 1): # +1 for system prompt
        # Keep system prompt + the latest N turns (user + assistant pairs)
        conversation_history = [conversation_history[0]] + conversation_history[-(MAX_HISTORY_TURNS * 2):]

    print(f"系统: 正在思考 (使用 {OLLAMA_MODEL})...")
    try:
        # Send the complete message history to Ollama
        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=conversation_history,
            stream=False # Get the full response at once
        )
        assistant_response = response['message']['content'].strip()

        # Add assistant's response to the history
        conversation_history.append({"role": "assistant", "content": assistant_response})

        print(f"LLM: {assistant_response}")
        return assistant_response

    except ollama.ResponseError as e:
        print(f"Ollama API 错误: {e.error}")
        print(f"Status Code: {e.status_code}")
        # Remove the user input that caused the error from history
        conversation_history.pop()
        return "抱歉，我在思考时遇到了API问题。" # More specific error
    except Exception as e:
        print(f"与 Ollama 通信时发生错误: {e}")
        # Remove the user input that caused the error from history
        if conversation_history and conversation_history[-1]["role"] == "user":
            conversation_history.pop()
        return "抱歉，连接语言模型时出现了一个错误。"


# --- TTS (ChatTTS) Function ---
def speak_chattts(text):
    """Synthesizes text using ChatTTS and plays the audio."""
    global chat_tts
    if not chat_tts:
        print("错误: ChatTTS 未初始化。")
        return
    if not text or not isinstance(text, str) or not text.strip():
        # print("系统: 无有效文本可供朗读。") # Optional info message
        return

    print(f"系统: 正在合成语音 (使用 ChatTTS on {CHATTS_DEVICE})...")
    try:
        # Basic inference with ChatTTS
        # Using use_decoder=True often yields better quality synthesis
        # Input text should be a list of strings
        wavs = chat_tts.infer([text.strip()], use_decoder=True)

        # wavs is a list of numpy arrays. We expect one result for one input string.
        if wavs and len(wavs) > 0:
            audio_data = wavs[0]
            if audio_data is not None and audio_data.size > 0:
                print(f"系统: 正在播放 (采样率: {CHATTS_SAMPLE_RATE} Hz)...")
                sd.play(audio_data, samplerate=CHATTS_SAMPLE_RATE)
                sd.wait() # Wait for the audio playback to complete
                print("系统: 播放完毕。")
            else:
                print("错误: ChatTTS 合成失败，返回的音频数据无效。")
        else:
             print("错误: ChatTTS 推理未返回有效的波形数据。")

    except Exception as e:
        print(f"ChatTTS 合成或播放错误: {e}")
        import traceback
        traceback.print_exc() # Print full stack trace for detailed debugging


# --- Initialization ---
def initialize():
    """Initializes Vosk, ChatTTS, Ollama, and the audio stream."""
    global vosk_recognizer, chat_tts, ollama_client, conversation_history, audio_stream

    print("--- 系统初始化 ---")
    initialized_ok = True

    # 1. Initialize Vosk STT
    print(f"正在加载 Vosk 模型 '{VOSK_MODEL_PATH}'...")
    if not os.path.exists(VOSK_MODEL_PATH):
        print(f"错误: Vosk 模型路径不存在 '{VOSK_MODEL_PATH}'")
        print("请确保模型已下载并放置在正确位置。")
        return False # Critical failure
    try:
        vosk_model = vosk.Model(VOSK_MODEL_PATH)
        vosk_recognizer = vosk.KaldiRecognizer(vosk_model, VOSK_SAMPLE_RATE)
        vosk_recognizer.SetWords(False) # Set to True if you need word timestamps
        print("Vosk 模型加载成功。")
    except Exception as e:
        print(f"加载 Vosk 模型失败: {e}")
        initialized_ok = False

    # 2. Initialize ChatTTS TTS
    if initialized_ok:
        print(f"正在加载 ChatTTS 模型 (Source: {CHATTS_MODEL_SOURCE}, Device: {CHATTS_DEVICE})...")
        if CHATTS_DEVICE == "cpu":
            print("警告: 在 CPU 上运行 ChatTTS 推理可能会比较慢。")
        try:
            # Dynamically import ChatTTS to avoid error if not installed
            from ChatTTS import Chat
            chat_tts = Chat()
            # Load models using the specified source

            # --- 修改在这里 ---
            if CHATTS_MODEL_SOURCE == 'local':
                print(f"  -> 从本地默认路径加载 (通常是 ./asset)")
                # 移除 local_path 参数
                chat_tts.load(source='local', device=CHATTS_DEVICE, force_redownload=False)
                # 重要提示: 确保你的模型文件确实在 ./asset 目录下 (相对于你运行脚本的位置)
            else:  # Assumes 'huggingface' or other source handled by ChatTTS library
                print(f"  -> 从 '{CHATTS_MODEL_SOURCE}' 加载/下载...")
                chat_tts.load(source=CHATTS_MODEL_SOURCE, device=CHATTS_DEVICE, force_redownload=False)
            # --- 修改结束 ---

            print("ChatTTS 模型加载成功。")
        except ImportError:
            print("\n错误: 未找到 'ChatTTS' 库。")
            print("请运行 'pip install chattts' 来安装。")
            initialized_ok = False
        except FileNotFoundError as fnf_error:
            # 这个错误现在可能在移除 local_path 后更容易出现，如果 asset 目录不存在或不完整
            print(f"\n加载 ChatTTS 模型失败 (文件未找到): {fnf_error}")
            print(f"请确认默认本地路径 (通常是 './asset') 下包含所需的 ChatTTS 文件。")
            print("如果模型文件不完整或位置不对，请检查或尝试让库重新下载 (可能需要临时设置 source='huggingface')。")
            initialized_ok = False
        except Exception as e:
            print(f"\n加载 ChatTTS 模型失败: {e}")
            print("-------------------- TRACEBACK --------------------")
            import traceback
            traceback.print_exc()  # Print full trace for detailed debugging
            print("---------------------------------------------------")
            initialized_ok = False

    # 3. Initialize Ollama LLM Client & Check Connection/Model
    if initialized_ok:
        print(f"正在连接 Ollama ({OLLAMA_HOST}) 并检查模型 '{OLLAMA_MODEL}'...")
        try:
            ollama_client = ollama.Client(host=OLLAMA_HOST)
            # Verify connection and model availability
            available_models_info = ollama_client.list() # Get list of dicts
            available_model_names = [m['name'] for m in available_models_info['models']]

            # Check if the exact model name or a name starting with it exists
            model_found = any(m_name == OLLAMA_MODEL or m_name.startswith(OLLAMA_MODEL.split(':')[0]) for m_name in available_model_names)

            if not model_found:
                 print(f"\n错误: 模型 '{OLLAMA_MODEL}' 未在 Ollama 中找到。")
                 print(f"请先运行 'ollama pull {OLLAMA_MODEL}' 并确保 Ollama 服务正在运行。")
                 print(f"当前可用模型: {available_model_names if available_model_names else '无'}")
                 initialized_ok = False
            else:
                 print(f"Ollama 连接成功，模型 '{OLLAMA_MODEL}' 可用。")
                 # Initialize conversation history with the system prompt
                 conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

        except ImportError:
             print("\n错误: 未找到 'ollama' 库。")
             print("请运行 'pip install ollama' 来安装。")
             initialized_ok = False
        except Exception as e:
            print(f"\n连接 Ollama 或检查模型时出错: {e}")
            print("请确保 Ollama 服务正在运行并且网络连接正常。")
            initialized_ok = False

    # 4. Start Audio Input Stream (Only if all previous steps succeeded)
    if initialized_ok:
        try:
            print("正在打开麦克风...")
            # Check available devices if needed: print(sd.query_devices())
            audio_stream = sd.InputStream(
                device=DEVICE_ID,         # Use default device or specify ID
                samplerate=VOSK_SAMPLE_RATE, # Must match Vosk model's sample rate
                channels=1,               # Mono audio
                dtype='int16',            # Data type expected by Vosk
                blocksize=BLOCK_SIZE,     # Size of audio chunks
                callback=audio_callback   # Function to handle incoming audio
            )
            audio_stream.start() # Start capturing audio in a background thread
            print("麦克风已打开，准备接收语音。")
        except Exception as e:
            print(f"打开麦克风失败: {e}")
            print("请检查是否有可用的麦克风以及 sounddevice 库是否正确安装。")
            initialized_ok = False

    # Final status report
    if initialized_ok:
        print("--- 初始化成功 ---")
    else:
        print("--- 初始化失败 ---")
        # Clean up partially initialized resources if needed (e.g., stop stream if started)
        if audio_stream and audio_stream.active:
            audio_stream.stop()
            audio_stream.close()
            print("麦克风流已关闭。")
    return initialized_ok

# --- Main Interaction Loop ---
if __name__ == "__main__":
    if not initialize():
        sys.exit(1) # Exit if initialization failed

    # Use the selected TTS function
    speak_func = speak_chattts
    speak_func("你好！本地语音对话系统已启动，请问有什么可以帮您的吗？")

    try:
        while True:
            # 1. Listen for user input
            user_input = listen_vosk()

            if user_input == "EXIT_SIGNAL": # Handle clean exit from listen_vosk (Ctrl+C)
                 print("\n系统: 检测到退出信号...")
                 break
            elif user_input:
                # Check if the user wants to exit
                if any(phrase in user_input for phrase in EXIT_PHRASES):
                    response = "好的，对话结束。下次再见！"
                    print(f"系统: {response}")
                    speak_func(response)
                    break

                # 2. Get response from LLM
                llm_response = get_llm_response(user_input)

                # 3. Speak the response
                if llm_response:
                    speak_func(llm_response)
                else:
                    # Handle case where LLM failed to generate a response
                    speak_func("抱歉，我处理您的问题时遇到了一些内部麻烦，暂时无法回答。")
            else:
                # Handle case where STT failed (no text recognized)
                # Optional: Provide feedback like "没听清"
                # speak_func("抱歉，我好像没有听清楚，您能再说一遍吗？")
                time.sleep(0.5) # Short pause before listening again to avoid immediate re-trigger

    except KeyboardInterrupt:
        print("\n系统: 检测到键盘中断 (Ctrl+C)... 正在退出...")
    finally:
        # --- Cleanup ---
        print("\n系统: 正在关闭...")
        # Stop and close the audio stream gracefully
        if audio_stream and audio_stream.active:
            try:
                recording_event.clear() # Ensure callback stops queueing
                audio_stream.stop()
                audio_stream.close()
                print("麦克风流已成功关闭。")
            except Exception as e:
                print(f"关闭麦克风流时出错: {e}")

        # Optional: Speak a final goodbye message
        # if chat_tts: # Check if TTS was initialized successfully
        #     speak_func("系统正在关闭，再见！")
        #     time.sleep(1) # Allow time for the final message to play

        print("系统: 程序结束。")