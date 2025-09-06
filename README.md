# Naru AI 🤖✨ - A Voice-Activated Conversational Assistant

Naru is a highly responsive, voice-activated AI assistant with a unique Hinglish personality. Built with Python, Groq, and state-of-the-art speech technologies, Naru is designed to be a helpful, context-aware, and engaging companion.



---

## 🚀 Features

* **Voice-Activated:** Hands-free operation using wake words ("Hey Naru").
* **High-Speed Responses:** Powered by the **Groq LPU™ Inference Engine** with the Llama 3.1 model for near-instantaneous replies.
* **Unique Personality:** A custom "Hinglish" persona that makes interactions feel natural and fun, not robotic.
* **Context-Aware:** Knows the current time, date, and your location to provide relevant answers.
* **Real-time Weather:** Integrated with Open-Meteo for live weather forecasts.
* **Noise Reduction:** Cleans up microphone input for better speech recognition in noisy environments.
* **Switchable Voices:** Choose between a male or female voice on the fly.
* **Conversation Memory:** Remembers the recent parts of your conversation for follow-up questions.
* **Extensible & Versatile:** Capable of handling a wide range of tasks:
    * Movie Recommendations
    * Finding Places
    * Human Emotion Improvement (empathetic conversation)
    * Weather Forecasts
    * Fashion Sensing and Advice
    * General Talk & Knowledge Q&A

---

## 🛠️ How It Works

Naru operates on a simple yet effective loop: **Listen -> Process -> Think -> Respond**.

1.  **Wake Word Detection:** Listens passively for a wake word (e.g., "Hey Naru").
2.  **Command Recording:** Upon activation, it records the user's command.
3.  **Audio Processing:** The recorded audio is cleaned using a noise reduction algorithm.
4.  **Speech-to-Text (STT):** The clean audio is transcribed into text using Google's STT engine.
5.  **Context Assembly:** The system gathers the user's query, conversation history, time, and real-time weather data.
6.  **LLM Inference:** The context-rich prompt is sent to the Groq API to generate a smart, in-character response.
7.  **Text-to-Speech (TTS):** The response text is converted into high-quality, natural-sounding audio using Microsoft Edge's neural voices.
8.  **Audio Playback:** The final audio is played back to the user.

---

## 🔧 Tech Stack

* **LLM Engine:** [Groq](https://groq.com/) (Llama 3.1 8B Instant)
* **Speech-to-Text:** `speech_recognition` (Google STT)
* **Text-to-Speech:** `edge-tts`
* **Audio Processing:** `pyaudio`, `pydub`, `sounddevice`, `noisereduce`
* **APIs:** Open-Meteo (Weather), TMDB (Movies)
* **Core Language:** Python 3.9+
* **Concurrency:** `asyncio`

---

## ⚙️ Setup and Installation

### Prerequisites

* Python 3.9+
* API Keys for:
    * [GroqCloud](https://console.groq.com/keys)
    * [TheMovieDB (TMDB)](https://www.themoviedb.org/settings/api) (Optional but recommended)
* System dependencies for PyAudio.
    * **On Debian/Ubuntu/Raspberry Pi OS:**
        ```bash
        sudo apt-get update && sudo apt-get install portaudio19-dev
        ```
    * **On macOS:**
        ```bash
        brew install portaudio
        ```

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/naru-ai.git](https://github.com/your-username/naru-ai.git)
    cd naru-ai
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    * Create a file named `.env` in the root directory.
    * Add your API keys to this file.

    **`.env` file:**
    ```env
    GROQ_API_KEY="gsk_YourGroqApiKeyHere"
    TMDB_API_KEY="YourTmdbApiKeyHere"
    ```

### requirements.txt

Your `requirements.txt` file should contain:

```txt
groq
requests
speechrecognition
edge-tts
python-dotenv
pydub
numpy
sounddevice
pyaudio
scipy
noisereduce
