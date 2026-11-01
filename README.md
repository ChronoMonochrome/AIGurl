# 🎙️ Virtual Girlfriend (AIGurl) — Russian Voice Assistant

A real-time, low-latency Russian voice assistant with personality generation via **Groq (Qwen models)** and natural emotional voice cloning using **CosyVoice 3.0 0.5B** (`Fun-CosyVoice3-0.5B`).

Designed with an advanced split architecture: a lightweight local Linux client managing full-duplex WebSocket audio streaming, VAD-based voice activity detection, microphone echo-locks, and a high-performance remote GPU inference server (Google Colab / FastAPI).

## ✨ Features

- **Dynamic Conversational Character:** Powered by Groq LLM (`qwen3.8-27b`) configured with a lively, opinionated, and modern persona (slang, sarcasm, and realism instead of robotic sugar-coating).
- **Full-Duplex WebSocket Streaming:** Real-time bi-directional communication between the local client and the Colab GPU server.
- **Context-Aware Speech Recognition (Whisper):** Integrates dynamic conversation history into OpenAI Whisper (`small` model) via `initial_prompt` to dramatically reduce transcription errors for slang, gaming terms, and conversational context.
- **Smart Echo & Interrupt Mitigation:** Client-side assistant speech locks (`assistant_is_speaking`) prevent the microphone from capturing speaker feedback, with instant buffer-clearing (`clear_buffer`) on interruption.
- **Expressive Voice Cloning & Inflection (CosyVoice 3.0 0.5B):** Utilizes Alibaba's state-of-the-art `Fun-CosyVoice3-0.5B` for zero-shot voice cloning, supporting multiple emotional styles (excited, sad, ironic, flirty, angry, passionate, etc.) and native high-quality speech synthesis.
- **Split Architecture for Low Latency:**
  * **Client (`main.py`):** Manages `sounddevice` microphone streaming, pause detection, multi-threaded audio downloading, and playback via `mpv`.
  * **TTS Server (Google Colab):** Handles WebSocket orchestration, Whisper transcription, and CosyVoice 3.0 0.5B zero-shot inference.
- **Configurable Environment:** Supports dynamic static ngrok tunnels configured via `.env` (`COLAB_URL`) and local debugging overrides (`--local`).

## 🛠️ Project Structure

```text
AIGurl/
├── main.py                # Local client (Groq API + WebSocket stream + Audio loop)
├── .env                   # Environment variables (GROQ_API_KEY, COLAB_URL)
├── requirements.txt       # Python dependencies
└── README.md              # Documentation

```

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ and `mpv` installed on your system.

```bash
# Install mpv audio player (Debian/Ubuntu)
sudo apt update && sudo apt install -y mpv

# Clone repo and create virtual environment
git clone https://github.com/ChronoMonochrome/AIGurl
cd AIGurl
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

```

### 2. Environment Setup

Create a `.env` file in the project root. **Note:** A static Ngrok URL (`COLAB_URL`) is required for stable WebSocket and HTTP connection between your client and the Google Colab server:

```env
GROQ_API_KEY=your_groq_api_key_here
COLAB_URL=https://your-static-ngrok-domain.ngrok-free.dev

```

## 💻 Usage

### Production Mode (Google Colab GPU Server)

1. Run the backend script in your Google Colab GPU environment (utilizing FastAPI, CosyVoice 3.0 0.5B, and Whisper) configured with your static ngrok domain.
2. Start the local client:
*For full interactive voice mode (WebSocket duplex):*
```bash
python main.py

```



*Optional CLI Arguments:*

* `--local`: Switch connection to a local server on `http://localhost:8000`.
* `--max-sentences N`: Limit maximum sentences generated per turn (defaults to `1` for ultra-low latency).
* `--prompt "text"`: Send a single text prompt instead of using the microphone.
* `--ref "emotion"`: Force a specific emotional reference/voice style (`excited`, `sad`, `ironic`, `flirty`, `surprised`, `angry`, `tired`, `passionate`, etc.).

