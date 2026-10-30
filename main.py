import asyncio
import os
import re
import subprocess
import threading
import time
from dotenv import load_dotenv
import edge_tts
import httpx
from groq import Groq

# 1. Загружаем переменные окружения
load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
  raise ValueError(
      "❌ Ошибка: Не найден GROQ_API_KEY. Проверьте файл .env."
  )

# Настраиваем прокси для edge-tts и groq
os.environ["http_proxy"] = "socks5://localhost:2080"
os.environ["https_proxy"] = "socks5://localhost:2080"

http_client = httpx.Client(
    proxy="socks5://localhost:2080",
    timeout=15.0,
)
client = Groq(api_key=api_key, http_client=http_client)

PUNCTUATION_PATTERN = re.compile(r"([.!?\n]+)")

# Словарь для хранения готовых файлов по индексам и мьютекс для потокобезопасности
audio_ready = {}
audio_lock = threading.Lock()


def clean_text_for_tts(text):
  emoji_pattern = re.compile(
      "["
      "\U0001f000-\U0001f6ff"
      "\U0001f900-\U0001f9ff"
      "\U0001f1e0-\U0001f6ff"
      "\u2600-\u26ff"
      "\u2700-\u27bf"
      "]+",
      flags=re.UNICODE,
  )
  text = emoji_pattern.sub(r"", text)
  text = text.replace("«", "").replace("»", "").replace("—", ",")
  return text.strip()


async def generate_audio_file(text, filename):
  communicate = edge_tts.Communicate(text, "ru-RU-SvetlanaNeural")
  await communicate.save(filename)


def download_worker(index, sentence, filename, t_created):
  """Параллельное скачивание конкретного предложения"""
  try:
    t_start = time.perf_counter()
    asyncio.run(generate_audio_file(sentence, filename))
    tts_duration = time.perf_counter() - t_start

    with audio_lock:
      audio_ready[index] = {
          "filename": filename,
          "sentence": sentence,
          "t_created": t_created,
          "tts_duration": tts_duration,
      }
  except Exception as e:
    print(f"⚠️ Ошибка скачивания TTS для фразы #{index}: {e}")


def playback_worker(total_sentences):
  """Воспроизведение строго по порядку (index 0, 1, 2...)"""
  for i in range(total_sentences):
    # Ждем, пока нужное предложение скачается
    while True:
      with audio_lock:
        if i in audio_ready:
          data = audio_ready.pop(i)
          break
      time.sleep(0.05)

    filename = data["filename"]
    sentence = data["sentence"]
    t_created = data["t_created"]
    tts_duration = data["tts_duration"]

    queue_wait_duration = time.perf_counter() - t_created

    t_play_start = time.perf_counter()
    subprocess.run(
        ["mpv", "--no-video", "--really-quiet", filename],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    play_duration = time.perf_counter() - t_play_start

    print(
        f"⏱️ [Метрика] Ожидание порядка: {queue_wait_duration:.2f}с |"
        f" Edge-TTS запрос: {tts_duration:.2f}с | Воспроизведение:"
        f" {play_duration:.2f}с | Текст: '{sentence}'"
    )

    if os.path.exists(filename):
      os.remove(filename)


def get_active_model():
  try:
    models_response = client.models.list()
    excluded = [
        "guard",
        "whisper",
        "embed",
        "safeguard",
        "classifier",
        "orpheus",
    ]
    all_models = [m.id for m in models_response.data]

    for model_id in all_models:
      if "qwen" in model_id.lower() and not any(
          ex in model_id.lower() for ex in excluded
      ):
        return model_id

    for model_id in all_models:
      if not any(ex in model_id.lower() for ex in excluded):
        return model_id
  except Exception as e:
    print(f"⚠️ Не удалось получить список моделей: {e}")

  return "qwen/qwen3.8-27b"


def stream_to_tts(prompt_text):
  model_name = get_active_model()
  print(f"⚡ Запуск генерации через модель [{model_name}]...")

  t_start = time.perf_counter()

  try:
    stream = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты — милая диалоговая виртуальная девушка. Отвечай"
                    " разговорным русским языком, кратко, вежливо и без"
                    " смайликов и эмодзи в тексте."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.7,
        stream=True,
    )

    buffer = ""
    sentences = []

    for chunk in stream:
      delta = chunk.choices[0].delta
      if delta and delta.content:
        buffer += delta.content

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

    total_sentences = len(sentences)
    print(
        f"📝 Сгенерировано предложений: {total_sentences}. Параллельная"
        " загрузка..."
    )

    # Очищаем словарь перед новым запуском
    audio_ready.clear()

    # Запускаем поток упорядоченного воспроизведения
    player_thread = threading.Thread(
        target=playback_worker, args=(total_sentences,), daemon=True
    )
    player_thread.start()

    # Запускаем параллельное скачивание с сохранением индексов
    download_threads = []
    for i, sentence in enumerate(sentences):
      filename = f"audio_part_{i}.mp3"
      t_created = time.perf_counter()
      d_thread = threading.Thread(
          target=download_worker, args=(i, sentence, filename, t_created)
      )
      d_thread.start()
      download_threads.append(d_thread)

    for d_thread in download_threads:
      d_thread.join()

    player_thread.join()

    total_time = time.perf_counter() - t_start
    print(f"\n✅ Общее время выполнения диалога: {total_time:.2f}с")

  except Exception as e:
    print(f"\n❌ Ошибка при обращении к Groq API: {e}")


if __name__ == "__main__":
  test_prompt = "Привет! Расскажи короткую историю о том, как ты провела день."
  stream_to_tts(test_prompt)
