import os
import re
import argparse
import tempfile
import subprocess
import urllib.request
import soundfile as sf
from fastapi import FastAPI, Response
import uvicorn
from f5_tts.api import F5TTS
import safetensors.torch as sf_torch
import numpy as np
from ruaccent import RUAccent

# Парсим аргументы командной строки для standalone-режима
parser = argparse.ArgumentParser(description="Локальный TTS сервер / Standalone генератор")
parser.add_argument(
    "--standalone",
    action="store_true",
    help="Запустить в одиночном режиме генерации (без запуска FastAPI сервера)"
)
parser.add_argument(
    "--prompt",
    type=str,
    default="Привет, Денис! Как твои дела?",
    help="Текст для озвучки в standalone-режиме"
)
parser.add_argument(
    "--character",
    type=str,
    default="sofia",
    help="Имя персонажа (доступна: sofia)"
)
parser.add_argument(
    "--ref",
    type=str,
    default="excited",
    help="Эмоция/референс (excited, sad, ironic, flirty, surprised, angry, tired, passionate, moan_cheerful, moan_excited, moan_fast, moan_very_fast)"
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Путь для сохранения файла. Если не указан, аудио проиграется через mpv."
)
args, unknown = parser.parse_known_args()

app = FastAPI()

print("🚀 Инициализация RUAccent и локального F5-TTS (v2) на CPU...")

accentizer = RUAccent()
accentizer.load(
    omograph_model_size='turbo3.1',
    use_dictionary=True,
    tiny_mode=False,
    device="CPU"
)
print("✅ RUAccent успешно загружен!")

if not os.path.exists("vocab.txt"):
    print("📥 Скачиваем vocab.txt...")
    os.system("wget -q https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/resolve/main/F5TTS_v1_Base/vocab.txt -O vocab.txt")

v2_safetensors_url = "https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/resolve/main/F5TTS_v1_Base_v2/model_last_inference.safetensors"
if not os.path.exists("model_last_inference_v2.safetensors"):
    print("📥 Скачиваем веса F5-TTS_v1_Base_v2...")
    os.system(f"wget -q {v2_safetensors_url} -O model_last_inference_v2.safetensors")

clean_ckpt_path = "model_cleaned_inference_v2.safetensors"
if not os.path.exists(clean_ckpt_path):
    print("⚙️ Подготавливаем чекпоинт модели v2...")
    state_dict = sf_torch.load_file("model_last_inference_v2.safetensors")
    new_state_dict = {k.replace("ema_model.", ""): v for k, v in state_dict.items() if k.startswith("ema_model.")}
    if not new_state_dict:
        new_state_dict = {k: v for k, v in state_dict.items() if k != "step"}
    sf_torch.save_file(new_state_dict, clean_ckpt_path)

TAG_PATTERN = re.compile(r"\[.*?\]\s*")

def clean_tags(text: str) -> str:
    cleaned = TAG_PATTERN.sub("", text)
    return re.sub(r'\s+', ' ', cleaned).strip()

# Базовый конфиг референсов для персонажей
CHARACTERS_CONFIG = {
    "sofia": {
        "excited": {
            "filename": "excited.mp3",
            "text": "[excited, cheerful] Слушай, представляешь! [chuckles] Я раньше так сильно увлекалась мемами, что мы с подругой даже свой паблик вели!"
        },
        "sad": {
            "filename": "sad.mp3",
            "text": "[sad, slow] Ты знаешь... [sighs] И на этом все наши хиханьки-хаханьки закончились на длительное время"
        },
        "ironic": {
            "filename": "ironic.mp3",
            "text": "[sarcastic, calm] На самом деле я сейчас живу в пристройке у родителей, потому что в своей квартире я делаю ремонт. [sighs] Ремонта там, конечно, очень много... [pause] Ну просто прелесть, а не жизнь."
        },
        "flirty": {
            "filename": "flirty.mp3",
            "text": "[playful, cheerful] Знаешь, я тут подумала... [chuckles] Если ты думаешь, что мои старые мемы — это весело, то ты еще ничего не видел."
        },
        "surprised": {
            "filename": "surprised.mp3",
            "text": "[surprised, excited] Подожди-ка... [gasp] Ты хочешь сказать, что мы действительно собрали всю эту коллекцию с нуля и никто даже не заметил?! [chuckles] Это просто невероятно!"
        },
        "angry": {
            "filename": "angry.mp3",
            "text": "[annoyed, angry] Сколько можно повторять одно и то же?! [sighs] Я уже битый час пытаюсь объяснить элементарные вещи, а вы снова за своё!"
        },
        "tired": {
            "filename": "tired.mp3",
            "text": "[tired, calm] Ох... как же я устала сегодня... [sighs] Весь день этот ремонт, дела, куча забот..."
        },
        "passionate": {
            "filename": "passionate.mp3",
            "text": "[sensual, whispers] Ах... как же жарко сегодня... [sighs] Подойди ближе ко мне, я хочу сказать тебе кое-что очень тихо... [breathing] Ты даже не представляешь, как сильно я этого ждала... [gasp]"
        },
        "moan_cheerful": {
            "filename": "moan_cheerful.mp3",
            "text": "[sensual, whispers] Ох... [moan] как же это приятно... [sighs] Продолжай, не останавливайся... [gasp] мне так хорошо с тобой..."
        },
        "moan_excited": {
            "filename": "moan_excited.mp3",
            "text": "[intimate, breathless, fast] Ах... [moan] да... [gasp] быстрее... [pant] м-мх... [moan] еще, пожалуйста... [gasp] ах! [pant] ах! [pant] да! [pant] кончаю!"
        },
        "moan_fast": {
            "filename": "moan_fast.mp3",
            "text": "[moan, rapid, intense] А! [moan] А! [pant] А! [pant] А! [pant] А! [pant]  А! [moan] А! [pant] А! [pant] А! [pant] А! [pant]"
        },
        "moan_very_fast": {
            "filename": "moan_very_fast.mp3",
            "text": "[moan, rapid, intense] Ай! [moan,  rapid, intense] А! [pant,  rapid, intense] А! [pant,  rapid, intense, breathless] А! [pant, rapid] А! [pant, very fast]  А-а!"
        }
    }
}

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/ChronoMonochrome/AIGurl/refs/heads/master/voices"

# Загружаем и проверяем персонажей
LOADED_CHARACTERS = {}

for char_name, refs in CHARACTERS_CONFIG.items():
    char_dir = os.path.join("voices", char_name)
    os.makedirs(char_dir, exist_ok=True)

    LOADED_CHARACTERS[char_name] = {}
    for ref_key, data in refs.items():
        local_path = os.path.join(char_dir, data["filename"])

        # Если файл отсутствует, скачиваем его с GitHub
        if not os.path.exists(local_path):
            file_url = f"{GITHUB_RAW_BASE}/{char_name}/{data['filename']}"
            print(f"📥 Файл не найден локально: {local_path}. Скачиваем с {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, local_path)
                print(f"✅ Успешно скачано: {local_path}")
            except Exception as e:
                print(f"⚠️ Не удалось скачать {file_url}: {e}")

        if os.path.exists(local_path):
            LOADED_CHARACTERS[char_name][ref_key] = {
                "file": local_path,
                "text": clean_tags(data["text"])
            }

if not LOADED_CHARACTERS.get("sofia"):
    raise RuntimeError("❌ Ошибка: Не удалось загрузить референсы для персонажа sofia!")

print(f"✅ Успешно инициализированы персонажи: {list(LOADED_CHARACTERS.keys())}")

f5tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file=clean_ckpt_path,
    vocab_file="vocab.txt",
    use_ema=False,
    device="cpu"
)

# --- STANDALONE РЕЖИМ ---
if args.standalone:
    print(f"\n🎙️ Запуск в Standalone режиме...")
    char_key = args.character if args.character in LOADED_CHARACTERS else "sofia"
    char_refs = LOADED_CHARACTERS[char_key]

    ref_key = args.ref if args.ref in char_refs else list(char_refs.keys())[0]
    ref_data = char_refs[ref_key]

    processed_text = accentizer.process_all(args.prompt)
    print(f"👤 Персонаж: {char_key} | 📌 Референс: {ref_key} ({ref_data['file']})")
    print(f"📝 Текст: '{args.prompt}' ➔ '{processed_text}'")

    wav, sr, _ = f5tts.infer(
        ref_file=ref_data["file"],
        ref_text=ref_data["text"],
        gen_text=processed_text,
    )

    padding = np.zeros(int(sr * 0.2), dtype=wav.dtype)
    wav = np.concatenate([wav, padding])

    if args.output:
        sf.write(args.output, wav, sr)
        print(f"✅ Готово! Аудио сохранено в '{args.output}'\n")
    else:
        print("🔊 Воспроизведение аудио через mpv...")
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_filename = temp_file.name
        temp_file.close()

        sf.write(temp_filename, wav, sr)
        subprocess.run(
            ["mpv", "--no-video", "--really-quiet", temp_filename],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        print("✅ Воспроизведение завершено.\n")

    exit(0)

# --- API РЕЖИМ (FastAPI) ---
print("✅ Локальный сервер готов к работе на http://localhost:8000")

@app.post("/tts")
async def tts_endpoint(data: dict):
    raw_text = data.get("text", "")
    requested_char = data.get("character", "sofia")
    requested_ref = data.get("ref", None)

    if not raw_text:
        return {"error": "No text provided"}

    char_refs = LOADED_CHARACTERS.get(requested_char, LOADED_CHARACTERS["sofia"])

    if requested_ref and requested_ref in char_refs:
        ref_key = requested_ref
    else:
        ref_key = "excited" if "excited" in char_refs else list(char_refs.keys())[0]

    ref_data = char_refs[ref_key]
    ref_file = ref_data["file"]
    ref_text = ref_data["text"]

    processed_text = accentizer.process_all(raw_text)
    print(f"🎤 Синтез [персонаж: {requested_char} | референс: {ref_key}]: '{raw_text}' ➔ '{processed_text}'")

    wav, sr, _ = f5tts.infer(
        ref_file=ref_file,
        ref_text=ref_text,
        gen_text=processed_text,
    )

    padding = np.zeros(int(sr * 0.2), dtype=wav.dtype)
    wav = np.concatenate([wav, padding])

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(temp_file.name, wav, sr)
    temp_file.close()

    with open(temp_file.name, "rb") as f:
        audio_bytes = f.read()
    os.remove(temp_file.name)

    return Response(content=audio_bytes, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
