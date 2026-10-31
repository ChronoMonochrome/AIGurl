import os
import re
import argparse
import subprocess
import threading
import time
from dotenv import load_dotenv
import httpx
from groq import Groq

# Парсим аргументы командной строки
parser = argparse.ArgumentParser(description="Клиент для голосового ассистента Virtual Girlfriend")
parser.add_argument(
    "--local",
    action="store_true",
    help="Использовать локальный сервер TTS (http://localhost:8000) вместо Google Colab"
)
parser.add_argument(
    "--max-sentences",
    type=int,
    default=None,
    help="Максимальное количество фраз для генерации (в локальном режиме по умолчанию 1)"
)
parser.add_argument(
    "--prompt",
    type=str,
    default="Привет! Расскажи короткую историю о том, как ты провела день.",
    help="Текст сообщения для отправки ассистенту"
)
parser.add_argument(
    "--ref",
    type=str,
    default=None,
    help="Принудительная эмоция/референс (excited, sad, ironic, flirty, surprised, angry, tired, passionate, moan_cheerful, moan_excited, moan_fast, moan_very_fast)"
)
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
    if args.max_sentences is None:
        args.max_sentences = 1
    print(f"🛠️ Режим отладки: запуск с ЛОКАЛЬНЫМ сервером TTS (лимит фраз: {args.max_sentences})")
    http_client = httpx.Client(timeout=180.0)
else:
    COLAB_URL = ""
    print(f"🌐 Режим продакшена: сервер Colab ({COLAB_URL}, лимит фраз: {args.max_sentences if args.max_sentences is not None else 'нет'})")
    http_client = httpx.Client(
        proxy="socks5://localhost:2080",
        timeout=45.0,
    )

client = Groq(api_key=api_key, http_client=http_client)

PUNCTUATION_PATTERN = re.compile(r"([.!?\n]+)")
EMOTION_TAG_PATTERN = re.compile(r"^\[([a-zA-Z_]+)\]\s*", re.UNICODE)

audio_ready = {}
audio_lock = threading.Lock()
tunnel_semaphore = threading.Semaphore(2)

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

    accent_fixes = {
        "пролила": "уронила",
        "звонила": "позвонила",
        "красивее": "красивей",
        "красивое": "прекрасное"
    }
    for wrong, right in accent_fixes.items():
        text = text.replace(wrong, right)

    text = text.strip()

    lower_text = text.lower()
    if "привет" in lower_text and len(text.split()) <= 2:
        text = "Ой, привет тебе…"
    elif ("как ты" in lower_text or "устал" in lower_text) and len(text.split()) <= 3:
        text = "Ну как ты там, дорогой?"
    elif len(text.split()) < 3:
        text = "Мм, " + text + "…"

    if text and not text.endswith(('.', '!', '?', '…', ',')):
        text += "…"
    elif text.endswith('.'):
        text = text[:-1] + "…"

    return text + "   "

def download_worker(index, sentence, emotion, filename, t_created):
    """Запрос к TTS с передачей выбранной Qwen или принудительной эмоции"""
    with tunnel_semaphore:
        try:
            t_start = time.perf_counter()
            headers = {"Bypass-Tunnel-Reminder": "true"}

            payload = {"text": sentence}
            # Приоритет у ручного флага --ref, иначе берем эмоцию, которую угадал Qwen
            chosen_ref = args.ref if args.ref else emotion
            if chosen_ref:
                payload["ref"] = chosen_ref

            response = http_client.post(
                f"{COLAB_URL}/tts",
                json=payload,
                headers=headers,
                timeout=180.0 if args.local else 40.0
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
            else:
                print(f"⚠️ Ошибка от TTS сервера для фразы #{index}: {response.text}")
        except Exception as e:
            print(f"⚠️ Ошибка соединения с сервером для фразы #{index}: {e}")

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

        for model_id in all_models:
            if not any(ex in model_id.lower() for ex in excluded):
                return model_id
    except Exception as e:
        print(f"⚠️ Не удалось получить список моделей: {e}")

    return "qwen/qwen3.8-27b"

def stream_to_tts(prompt_text):
    model_name = get_active_model()
    print(f"⚡ Запуск генерации через модель [{model_name}] с промптом: «{prompt_text}»...")

    t_start = time.perf_counter()

    try:
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — живая виртуальная девушка. Твой собеседник — ПАРЕНЬ по имени Денис (обращайся к нему "
                        "в мужском роде: 'провел', 'устал', 'сделал', 'дорогой'). "
                        "Общайся современным, живым и естественным языком, как в переписке. "
                        "Отвечай не более 3-5 коротких предложений суммарно, но избегай очень коротких предложений. "
                        "КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО использовать книжные деепричастия ('видя', 'ощущая'). "
                        "Используй простые глаголы.\n\n"
                        "ВАЖНО: В самом начале своего ответа ты ОБЯЗАТЕЛЬНА должна указать эмоцию в квадратных скобках, "
                        "точно отражающую твое текущее состояние и реакцию на слова собеседника.\n"
                        "Доступные эмоции: [excited], [ironic], [flirty], [surprised], [angry], [tired], [sad], [passionate], "
                        "[moan_cheerful].\n\n"
                        "ПРАВИЛО ДЛЯ ИНТИМА: Если промпт интимный или содержит стоны/прикосновения, "
                        "ты ОБЯЗАТЕЛЬНА должна выбрать один из стонущих референсов ([moan_cheerful]) "
                        "и органично добавить легкие томные вздохи или стоны в текст ответа  "
                        "предложения не должны состоять из одних междометий, а чередовать АХи, Охи и слова"
                        "предложения должны быть длинными, сделай очень вариативный и интимный ответ"
                    ),
                },
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.85,
            stream=True,
        )

        buffer = ""
        sentences = []
        detected_emotion = "excited"  # дефолтная эмоция
        emotion_extracted = False

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                buffer += delta.content

                # Извлекаем тег эмоции из начала ответа LLM
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

        if args.max_sentences is not None and len(sentences) > args.max_sentences:
            if args.max_sentences == 1 and len(sentences) > 0:
                mid_idx = len(sentences) // 2
                sentences = [sentences[mid_idx]]
            else:
                sentences = sentences[:args.max_sentences]

        total_sentences = len(sentences)
        print(f"📝 Сгенерировано фраз для отправки: {total_sentences} (лимит: {args.max_sentences if args.max_sentences is not None else 'нет'}). Потоковая передача...")

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

        total_time = time.perf_counter() - t_start
        print(f"\n✅ Общее время диалога: {total_time:.2f}с")

    except Exception as e:
        print(f"\n❌ Ошибка при обращении к Groq API: {e}")

if __name__ == "__main__":
    stream_to_tts(args.prompt)
