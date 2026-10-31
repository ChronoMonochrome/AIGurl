# 🎙️ Virtual Girlfriend (AIGurl) — Russian Voice Assistant

A real-time, low-latency Russian voice assistant with personality generation via **Groq (Qwen models)** and natural expressive speech synthesis using **F5-TTS v2** with **RUAccent** stress processing.

Designed with a split architecture: a lightweight local Linux client managing stream processing and audio execution, and a high-performance remote GPU inference server (Google Colab / FastAPI).

## ✨ Features

* **Dynamic Conversational Character:** Powered by Groq LLM (`qwen3.8-27b`) configured with a casual, non-literary Russian persona for "Денис".
* **Expressive TTS with Stress Placement:** Utilizes the fine-tuned `F5-TTS_RUSSIAN` (v2 checkpoint) combined with `RUAccent` for automatic stress placement (`+` markers).
* **Dynamic Prosody & Padding:** Adaptive NFE diffusion steps based on phrase length (e.g., lower steps for short interjections like "хи-хи", "ой" to make them sound natural) and end-padding to eliminate clipping.
* **Split Architecture for Low Latency:**
  * **Client (`main.py`):** Parallelized streaming downloader & playback queue managing audio execution via `mpv`.
  * **TTS Server (`local_tts_server.py` / Colab):** Handles F5-TTS v2 inference, RUAccent stress placement, and audio padding.
* **Dual-Mode Operation:** 
  * Default production mode uses Colab GPU with unlimited sentence streaming.
  * `--local` debug mode targets CPU server with an automatic 1-sentence limit (extracting the middle sentence for quick iteration).

## 🛠️ Project Structure

```
AIGurl/
├── main.py                     # Local client (Groq API + Streaming Playback)
├── local_tts_server.py         # Local CPU FastAPI server for debugging
├── ref.wav                     # Target voice audio reference
├── .env                        # Environment variables (Groq API keys)
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ and `mpv` installed on your system.

```bash
# Install mpv audio player (Debian/Ubuntu)
sudo apt update && sudo apt install -y mpv

# Clone repo and create virtual environment
git clone https://github.com/your-username/AIGurl.git
cd AIGurl
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Place your reference audio file in the project root as `ref.wav`.

## 💻 Usage

### Production Mode (Google Colab GPU Server)

1. Run the single-cell Colab script with your `NGROK_TOKEN` and `STATIC_URL`.
2. Start the local client (runs without sentence limits by default):

   ```bash
   python main.py
   ```

### Local Debugging Mode (CPU)

1. Launch the local FastAPI server on CPU:

   ```bash
   python local_tts_server.py
   ```

2. In a separate terminal, run the client pointing to localhost (automatically defaults to `--max-sentences 1` taking the middle sentence):

   ```bash
   python main.py --local
   ```

   *Optional CLI Arguments:*
   * `--local`: Switch to local TTS server on `http://localhost:8000`.
   * `--max-sentences N`: Manually override the maximum number of sentences processed per response (e.g., `python main.py --local --max-sentences 3`).
