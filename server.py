import os
import json
import tempfile
import urllib.request
import asyncio
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Response
import uvicorn
import torch
import torchaudio
import numpy as np
import whisper
from pyngrok import ngrok
import sys
import site
import nest_asyncio

nest_asyncio.apply()

# Исправление путей для cuDNN (для ONNX Runtime / GPU)
site_packages = site.getsitepackages()[0]
cudnn_lib = os.path.join(site_packages, 'nvidia', 'cudnn', 'lib')
cublas_lib = os.path.join(site_packages, 'nvidia', 'cublas', 'lib')
os.environ['LD_LIBRARY_PATH'] = f"{cudnn_lib}:{cublas_lib}:" + os.environ.get('LD_LIBRARY_PATH', '')

sys.path.append(os.path.abspath("."))
from cosyvoice.cli.cosyvoice import AutoModel

app = FastAPI()

print("🎙️ Инициализация Whisper (small) и Fun-CosyVoice3-0.5B на GPU...")
whisper_model = whisper.load_model("small", device="cuda")

# Используем модель Fun-CosyVoice3-0.5B
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
print("✅ Whisper и Fun-CosyVoice3 успешно загружены!")

CHARACTERS_CONFIG = {
    "sofia": {
        "excited": {"filename": "excited.mp3", "text": "Слушай, представляешь! Я раньше так сильно увлекалась мемами, что мы с подругой даже свой паблик вели!"},
        "sad": {"filename": "sad.mp3", "text": "Ты знаешь... И на этом все наши хиханьки-хаханьки закончились на длительное время."},
        "ironic": {"filename": "ironic.mp3", "text": "На самом деле я сейчас живу в пристройке у родителей, потому что в своей квартире я делаю ремонт. Ну просто прелесть, а не жизнь."},
        "flirty": {"filename": "flirty.mp3", "text": "Знаешь, я тут подумала..."},
        "surprised": {"filename": "surprised.mp3", "text": "Подожди-ка... Ты хочешь сказать, что мы действительно собрали всю эту коллекцию с нуля и никто даже не заметил?! Это просто невероятно!"},
        "angry": {"filename": "angry.mp3", "text": "Сколько можно повторять одно и то же?! Я уже битый час пытаюсь объяснить элементарные вещи, а вы снова за своё!"},
        "tired": {"filename": "tired.mp3", "text": "Ох... как же я устала сегодня... Весь день этот ремонт, дела, куча забот..."},
        "passionate": {"filename": "passionate.mp3", "text": "Ах... как же жарко сегодня... Подойди ближе ко мне, я хочу сказать тебе кое-что очень тихо... Ты даже не представляешь, как сильно я этого ждала..."},
        "moan_cheerful": {
            "filename": "moan_cheerful.mp3",
            "text": "Ох... как же это приятно... Продолжай, не останавливайся... мне так хорошо с тобой..."
        },
        "moan_excited": {
            "filename": "moan_excited.mp3",
            "text": "Ах... да... быстрее... м-мх... еще, пожалуйста... ах! ах! да! кончаю!"
        },
        "moan_fast": {
            "filename": "moan_fast.mp3",
            "text": "А! А! А! А! А! А! А! А! А! А!"
        },
        "moan_very_fast": {
            "filename": "moan_very_fast.mp3",
            "text": "Ай! А! А! А! А! А-а!"
        },
    }
}

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/ChronoMonochrome/AIGurl/refs/heads/master/voices"
LOADED_CHARACTERS = {}

print("📥 Проверка и подготовка кастомных референсов голосов...")
for char_name, refs in CHARACTERS_CONFIG.items():
    char_dir = os.path.join("voices", char_name)
    os.makedirs(char_dir, exist_ok=True)
    LOADED_CHARACTERS[char_name] = {}
    for ref_key, data in refs.items():
        mp3_path = os.path.join(char_dir, data["filename"])
        wav_filename = data["filename"].rsplit(".", 1)[0] + "_16k.wav"
        wav_path = os.path.join(char_dir, wav_filename)

        if not os.path.exists(mp3_path):
            file_url = f"{GITHUB_RAW_BASE}/{char_name}/{data['filename']}"
            try:
                print(f"📥 Скачивание {file_url}...")
                urllib.request.urlretrieve(file_url, mp3_path)
            except Exception as e:
                print(f"⚠️ Ошибка скачивания {file_url}: {e}")

        if os.path.exists(mp3_path) and not os.path.exists(wav_path):
            try:
                speech, sr = torchaudio.load(mp3_path)
                if speech.shape[0] > 1:
                    speech = speech.mean(dim=0, keepdim=True)
                if sr != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                    speech = resampler(speech)
                torchaudio.save(wav_path, speech, 16000)
                print(f"✅ Успешно подготовлен референс: {wav_path}")
            except Exception as e:
                print(f"⚠️ Ошибка обработки референса {mp3_path}: {e}")

        if os.path.exists(wav_path):
            LOADED_CHARACTERS[char_name][ref_key] = {
                "prompt_speech_16k": wav_path,
                "prompt_text": data["text"]
            }
        else:
            print(f"❌ Критично: файл референса не найден -> {wav_path}")

print("✅ Инициализация референсов завершена.")

@app.post("/tts")
async def tts_endpoint(data: dict):
    raw_text = data.get("text", "")
    requested_char = data.get("character", "sofia")
    requested_ref = data.get("ref") or data.get("emotion") or data.get("style") or "excited"

    if not raw_text:
        return {"error": "No text provided"}

    char_refs = LOADED_CHARACTERS.get(requested_char, LOADED_CHARACTERS["sofia"])
    ref_data = char_refs.get(requested_ref)
    if not ref_data:
        ref_data = char_refs.get("excited", {})

    prompt_wav = ref_data.get("prompt_speech_16k")
    if not prompt_wav or not os.path.exists(prompt_wav):
        return {"error": f"Reference voice file not found on disk: {prompt_wav}"}, 400

    print(f"🎯 [TTS Запрос] Персонаж: {requested_char} | Эмоция: {requested_ref} | Текст: {raw_text}")

    try:
        audio_chunks = []
        # Если синтезируемый текст слишком короткий относительно промпта, не передаем prompt_text, чтобы избежать багов CosyVoice 3
        if len(raw_text.split()) < 4:
            prompt_text_with_instruct = "You are a helpful assistant.<|endofprompt|>"
        else:
            prompt_text_with_instruct = f"You are a helpful assistant.<|endofprompt|>{ref_data.get('prompt_text', '')}"

        for i, j in enumerate(cosyvoice.inference_zero_shot(
            raw_text,
            prompt_text_with_instruct,
            prompt_wav,
            stream=False
        )):
            speech_np = j['tts_speech'].detach().cpu().numpy()
            audio_chunks.append(speech_np)

        if not audio_chunks:
            return {"error": "TTS generation failed: empty output"}, 500

        wav = np.concatenate(audio_chunks, axis=-1)
        if wav.ndim > 1:
            wav = wav.squeeze()

        sr = cosyvoice.sample_rate
        padding = np.zeros(int(sr * 0.15), dtype=wav.dtype)
        wav = np.concatenate([wav, padding])

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wav, sr)
        temp_file.close()

        with open(temp_file.name, "rb") as f:
            audio_bytes = f.read()
        os.remove(temp_file.name)
        return Response(content=audio_bytes, media_type="audio/wav")

    except Exception as e:
        print(f"❌ Ошибка синтеза Fun-CosyVoice3: {e}")
        return {"error": str(e)}, 500

@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Клиент подключен по WebSocket (дуплекс)")

    audio_buffer = bytearray()
    transcription_history = []

    try:
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=35.0)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
                continue

            if "bytes" in message:
                audio_buffer.extend(message["bytes"])

            elif "text" in message:
                data = json.loads(message["text"])
                action = data.get("action")

                if action == "speech_end":
                    if len(audio_buffer) > 0:
                        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        temp_name = temp_audio.name
                        temp_audio.close()

                        audio_np = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                        sf.write(temp_name, audio_np, 16000)
                        audio_buffer.clear()

                        dynamic_prompt = "Разговор Дениса и девушки. " + " ".join(transcription_history[-6:])

                        print("🎙️ Распознавание речи через Whisper...")
                        result = whisper_model.transcribe(
                            temp_name,
                            language="ru",
                            initial_prompt=dynamic_prompt,
                            condition_on_previous_text=False,
                            beam_size=5
                        )
                        user_text = result.get("text", "").strip()

                        if os.path.exists(temp_name):
                            os.remove(temp_name)

                        print(f"📝 Распознанный текст: '{user_text}'")

                        if user_text:
                            transcription_history.append(f"Денис: {user_text}")

                        await websocket.send_json({
                            "type": "transcription",
                            "text": user_text
                        })
                elif action == "assistant_said":
                    bot_text = data.get("text", "")
                    if bot_text:
                        transcription_history.append(f"София: {bot_text}")
                elif action in ("clear_buffer", "interrupt"):
                    audio_buffer.clear()
                    print("⚡ Буфер аудио на сервере очищен.")

    except WebSocketDisconnect:
        print("🔌 Клиент отключился от WebSocket")
    except Exception as e:
        print(f"⚠️ Ошибка в WebSocket: {e}")

if __name__ == "__main__":
    NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")
    STATIC_URL = os.environ.get("STATIC_URL", "")

    try:
        if NGROK_TOKEN:
            ngrok.set_auth_token(NGROK_TOKEN)
    except:
        pass

    if STATIC_URL:
        public_url = ngrok.connect(8000, url=STATIC_URL).public_url
    else:
        public_url = ngrok.connect(8000).public_url

    print(f"\n🌐 Статический туннель ngrok успешно запущен: {public_url}\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
