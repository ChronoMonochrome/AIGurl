import os
import re
import argparse
import subprocess
import threading
import time
import asyncio
import json
import websockets
import sounddevice as sd
import numpy as np
from dotenv import load_dotenv
import httpx
from groq import Groq

parser = argparse.ArgumentParser(description="Клиент для голосового ассистента Virtual Girlfriend")
parser.add_argument("--local", action="store_true", help="Использовать локальный сервер TTS")
parser.add_argument("--max-sentences", type=int, default=1, help="Максимальное количество фраз")
parser.add_argument("--prompt", type=str, default=None, help="Текст сообщения")
parser.add_argument("--ref", type=str, default=None, help="Принудительная эмоция")
args = parser.parse_args()

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ Ошибка: Не найден GROQ_API_KEY. Проверьте файл .env.")

os.environ["http_proxy"] = "socks5://localhost:2080"
os.environ["https_proxy"] = "socks5://localhost:2080"
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

if args.local:
    COLAB_URL = "http://localhost:8000"
    WS_URL = "ws://localhost:8000/ws/voice"
    http_client = httpx.Client(timeout=1800.0)
else:
    COLAB_URL = os.environ.get("COLAB_URL")
    if not COLAB_URL:
        raise ValueError("❌ Ошибка: Не найден COLAB_URL в файле .env.")
    COLAB_URL = COLAB_URL.rstrip("/")
    WS_URL = COLAB_URL.replace("https://", "wss://").replace("http://", "ws://") + "/ws/voice"
    http_client = httpx.Client(proxy="socks5://localhost:2080", timeout=45.0)

client = Groq(api_key=api_key, http_client=http_client)

PUNCTUATION_PATTERN = re.compile(r"([.!?\n]+)")
EMOTION_TAG_PATTERN = re.compile(r"^\[([a-zA-Z_]+)\]\s*", re.UNICODE)

audio_ready = {}
audio_lock = threading.Lock()
tunnel_semaphore = threading.Semaphore(2)

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.5
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
SILENCE_THRESHOLD = 0.0008
PAUSE_LIMIT_BLOCKS = 3
is_recording = True
audio_queue = asyncio.Queue()
assistant_is_speaking = False

chat_history = [
    {
        "role": "system",
        "content": (
            "Ты — живая, характерная девушка со своими заморочками, а не послушный чат-бот. Твой парень — Денис. "
            "Общайся максимально живым, разговорным языком реального мессенджера (используй сленг, сокращения, спорь, иронизируй, иногда капризничай). "
            "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО: отвечать шаблонными сопливыми фразами вроде 'думаю о тебе', 'хочу обнять', 'скучаю' без повода. "
            "Будь живой: у тебя могут быть свои дела, плохое настроение, ты можешь подкалывать Дениса, отвлекаться на кота/сериал или спорить с ним. "
            "Отвечай с искрой. "
            "В начале ответа обязательно укажи эмоцию в квадратных скобках: "
            "[excited] [flirty], [surprised], [ironic], [angry], [tired], [sad], [passionate], [moan_fast], [moan_very_fast], [moan_cheerful], [moan_excited]."
            "ПРАВИЛО ДЛЯ ИНТИМА: Если контекст интимный, ты ДОЛЖНА использовать один из вариантов [moan_fast], [moan_very_fast], [moan_cheerful], [moan_excited]."
            "ЗАПРЕЩЕНО: СЛИШКОМ КОРОТКИЕ ПРЕДЛОЖЕНИЯ"
        ),
    }
]

def clean_text_for_tts(text):
    emoji_pattern = re.compile("["
        "\U0001f000-\U0001f6ff"
        "\U0001f900-\U0001f9ff"
        "\U0001f1e0-\U0001f6ff"
        "\u2600-\u26ff"
        "\u2700-\u27bf"
    "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r"", text)
    text = text.replace("*", "").replace("«", "").replace("»", "").replace("—", ",")
    text = text.strip()

    if text and not text.endswith(('.', '!', '?', '…', ',')):
        text += "…"
    elif text.endswith('.'):
        text = text[:-1] + "…"

    return text + "    "

def download_worker(index, sentence, emotion, filename, t_created):
    with tunnel_semaphore:
        try:
            t_start = time.perf_counter()
            headers = {"Bypass-Tunnel-Reminder": "true"}

            payload = {
                "text": sentence,
                "history": chat_history[-4:]  # Передаем контекст последних сообщений
            }
            chosen_ref = args.ref if args.ref else emotion
            if chosen_ref:
                payload["ref"] = chosen_ref

            response = http_client.post(
                f"{COLAB_URL}/tts",
                json=payload,
                headers=headers,
                timeout=4000.0
            )

            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                tts_duration = time.perf_counter() - t_start

                with audio_lock:
                    audio_ready[index] = {
                        "filename": filename,
                        "sentence": sentence,
                        "emotion": chosen_ref,
                        "t_created": t_created,
                        "tts_duration": tts_duration
                    }
        except Exception as e:
            print(f"⚠️ Ошибка TTS: {e}")

def playback_worker(total_sentences):
    for i in range(total_sentences):
        while True:
            with audio_lock:
                if i in audio_ready:
                    data = audio_ready.pop(i)
                    break
            time.sleep(0.03)

        filename = data["filename"]
        sentence = data["sentence"]
        emotion = data["emotion"]
        t_created = data["t_created"]
        tts_duration = data["tts_duration"]

        queue_wait_duration = time.perf_counter() - t_created

        t_play_start = time.perf_counter()
        subprocess.run(
            ["mpv", "--no-video", "--really-quiet", filename],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        play_duration = time.perf_counter() - t_play_start

        print(f"⏱️ [Метрика] Эмоция: [{emotion}] | Ожидание: {queue_wait_duration:.2f}с | Генерация: {tts_duration:.2f}с | Воспроизведение: {play_duration:.2f}с | Текст: '{sentence}'")

        if os.path.exists(filename):
            os.remove(filename)

def get_active_model():
    try:
        models_response = client.models.list()
        excluded = ["guard", "whisper", "embed", "safeguard", "classifier", "orpheus"]
        all_models = [m.id for m in models_response.data]
        for model_id in all_models:
            if "qwen" in model_id.lower() and not any(ex in model_id.lower() for ex in excluded):
                return model_id
    except Exception:
        pass
    return "qwen/qwen3.8-27b"

def stream_to_tts(prompt_text, websocket=None):
    global assistant_is_speaking, chat_history
    assistant_is_speaking = True

    model_name = get_active_model()
    print(f"⚡ Запуск генерации через модель [{model_name}] с промптом: «{prompt_text}»...")

    t_start = time.perf_counter()

    try:
        chat_history.append({"role": "user", "content": prompt_text})
        if len(chat_history) > 7:
            chat_history = [chat_history[0]] + chat_history[-6:]

        stream = client.chat.completions.create(
            model=model_name,
            messages=chat_history,
            temperature=0.7,
            stream=True,
        )

        buffer = ""
        sentences = []
        detected_emotion = "excited"
        emotion_extracted = False

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                buffer += delta.content

                if not emotion_extracted:
                    match_emo = EMOTION_TAG_PATTERN.search(buffer)
                    if match_emo:
                        detected_emotion = match_emo.group(1)
                        print(f"🎭 Qwen выбрала эмоцию: [{detected_emotion}]")
                        buffer = buffer[match_emo.end():]
                        emotion_extracted = True
                    elif len(buffer) > 20 and "[" not in buffer:
                        emotion_extracted = True

                while True:
                    match = PUNCTUATION_PATTERN.search(buffer)
                    if not match:
                        break

                    split_idx = match.end()
                    sentence = buffer[:split_idx].strip()
                    buffer = buffer[split_idx:]

                    if sentence:
                        cleaned = clean_text_for_tts(sentence)
                        if cleaned:
                            sentences.append(cleaned)

        if buffer.strip():
            cleaned = clean_text_for_tts(buffer.strip())
            if cleaned:
                sentences.append(cleaned)

        if not sentences:
            sentences = [clean_text_for_tts("Да, я тут, внимательно тебя слушаю…")]

        max_s = args.max_sentences if args.max_sentences is not None else 2
        sentences = sentences[:max_s]

        full_assistant_reply = "".join(sentences)
        chat_history.append({"role": "assistant", "content": full_assistant_reply})

        total_sentences = len(sentences)
        audio_ready.clear()

        player_thread = threading.Thread(target=playback_worker, args=(total_sentences,), daemon=True)
        player_thread.start()

        download_threads = []
        for i, sentence in enumerate(sentences):
            filename = f"audio_part_{i}.wav"
            t_created = time.perf_counter()
            d_thread = threading.Thread(target=download_worker, args=(i, sentence, detected_emotion, filename, t_created))
            d_thread.start()
            download_threads.append(d_thread)

        for d_thread in download_threads:
            d_thread.join()

        player_thread.join()
        print(f"\n✅ Ответ передан за {time.perf_counter() - t_start:.2f}с")

    except Exception as e:
        print(f"\n❌ Ошибка Groq: {e}")
    finally:
        if websocket is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(websocket.send(json.dumps({"action": "clear_buffer"})), loop)
                    future.result(timeout=2.0)
            except Exception:
                pass
        assistant_is_speaking = False
        print("🎙️ Микрофон снова активен.")

async def mic_stream_loop(websocket):
    loop = asyncio.get_running_loop()
    silent_blocks_count = 0
    is_speaking = False

    def callback(indata, frames, time_info, status):
        nonlocal silent_blocks_count, is_speaking
        if assistant_is_speaking:
            return

        mono_data = indata[:, 0] if indata.ndim > 1 and indata.shape[1] > 1 else indata.flatten()
        audio_int16 = (mono_data * 32767).astype(np.int16)
        volume = np.linalg.norm(mono_data) / len(mono_data)
        loop.call_soon_threadsafe(audio_queue.put_nowait, (audio_int16.tobytes(), volume))

    print("🎙️ Микрофон активен. Можете говорить...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', blocksize=BLOCK_SIZE, latency='high', callback=callback):
        while is_recording:
            chunk_bytes, volume = await audio_queue.get()

            if assistant_is_speaking:
                continue

            if volume > SILENCE_THRESHOLD:
                if not is_speaking:
                    print("🗣️ Речь обнаружена...")
                    is_speaking = True
                silent_blocks_count = 0
                await websocket.send(chunk_bytes)
            else:
                if is_speaking:
                    silent_blocks_count += 1
                    await websocket.send(chunk_bytes)

                    if silent_blocks_count >= PAUSE_LIMIT_BLOCKS:
                        print("🤫 Пауза зафиксирована. Отправка на распознавание...")
                        is_speaking = False
                        silent_blocks_count = 0

                        await websocket.send(json.dumps({"action": "speech_end"}))

                        try:
                            response_raw = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                            msg = json.loads(response_raw)
                            if msg.get("type") == "transcription":
                                user_text = msg.get("text", "")
                                if user_text:
                                    print(f"📝 Распознано от Whisper: «{user_text}»")
                                    threading.Thread(target=stream_to_tts, args=(user_text, websocket), daemon=True).start()
                        except asyncio.TimeoutError:
                            print("⚠️ Таймаут ожидания ответа")

async def main_ws():
    print(f"🔗 Подключение к WebSocket: {WS_URL}...")
    async with websockets.connect(WS_URL, additional_headers={"Bypass-Tunnel-Reminder": "true"}, ping_interval=20, ping_timeout=20) as websocket:
        print("✅ Успешное подключение!")
        await mic_stream_loop(websocket)

if __name__ == "__main__":
    if args.prompt is not None:
        stream_to_tts(args.prompt, None)
    else:
        try:
            asyncio.run(main_ws())
        except KeyboardInterrupt:
            print("\n⏹️ Завершено пользователем.")
