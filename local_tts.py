import os
import tempfile
import soundfile as sf
from fastapi import FastAPI, Response
import uvicorn
from f5_tts.api import F5TTS
import safetensors.torch as sf_torch
import numpy as np
from ruaccent import RUAccent  # 👈 Импортируем авторасстановку ударений

app = FastAPI()

print("🚀 Инициализация RUAccent и локального F5-TTS (v2) на CPU...")

# 1. Инициализируем RUAccent для автоматической расстановки ударений (+)
accentizer = RUAccent()
accentizer.load(
    omograph_model_size='turbo3.1',
    use_dictionary=True,
    tiny_mode=False,
    device="CPU"
)
print("✅ RUAccent успешно загружен!")

# 2. Скачиваем вокабуляр и веса актуальной версии v2 модели F5-TTS_RUSSIAN
if not os.path.exists("vocab.txt"):
    print("📥 Скачиваем vocab.txt...")
    os.system("wget -q https://huggingface.co/Misha24-10/F5-TTS_RUSSIAN/resolve/main/F5TTS_v1_Base/vocab.txt -O vocab.txt")

# Ссылка на веса версии v2 (модель v2)
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

REF_AUDIO = "ref.wav"
REF_TEXT = "На самом деле я живу не одна, я живу в пристройке у родителей"

if not os.path.exists(REF_AUDIO):
    print(f"⚠️ ВНИМАНИЕ: Не найден файл референса '{REF_AUDIO}'! Положите его в папку со скриптом.")

# Инициализируем F5-TTS с весами версии v2 (на CPU)
f5tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file=clean_ckpt_path,
    vocab_file="vocab.txt",
    use_ema=False,
    device="cpu"
)

print("✅ Локальный сервер готов к работе на http://localhost:8000")

@app.post("/tts")
async def tts_endpoint(data: dict):
    raw_text = data.get("text", "")
    if not raw_text:
        return {"error": "No text provided"}

    # Автоматически проставляем ударения через RUAccent (модель v2 обучена на текстах с плюсиками)
    processed_text = accentizer.process_all(raw_text)
    print(f"🎤 Синтез (v2 + ударения): '{raw_text}' ➔ '{processed_text}'")

    # Генерация через F5-TTS
    wav, sr, _ = f5tts.infer(
        ref_file=REF_AUDIO,
        ref_text=REF_TEXT,
        gen_text=processed_text,
    )

    # Добавляем 200 мс тишины в конец для плавного затухания
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
