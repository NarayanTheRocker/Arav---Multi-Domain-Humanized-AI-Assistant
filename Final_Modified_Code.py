import json
import requests
from groq import Groq
from datetime import datetime
import os
import speech_recognition as sr
import edge_tts
import asyncio
import io
from dotenv import load_dotenv
from pydub import AudioSegment
import RPi.GPIO as GPIO
import numpy as np
import sounddevice as sd
import time
import re
import wave
import pyaudio
import scipy.io.wavfile as wav
import noisereduce as nr

# --- Configuration and Constants ---
# Load environment variables from a .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# --- Assistant Settings ---
HISTORY_FILE = "conversation_history.json"
MAX_HISTORY = 30
WAKE_WORDS = ["arav", "ara", "aarav", "naru", "hey naru"]
CONSECUTIVE_SILENCE_THRESHOLD = 2 # Go back to sleep after this many failed listens

# --- Audio Recording Settings ---
MIC_SAMPLE_RATE = 16000  # 16kHz is standard for speech
MIC_CHANNELS = 1
MIC_FORMAT = pyaudio.paInt16
MIC_CHUNK_SIZE = 1024
WAKE_WORD_RECORD_SECONDS = 3
COMMAND_RECORD_SECONDS = 7
RAW_AUDIO_FILENAME = "temp_raw_audio.wav"
CLEAN_AUDIO_FILENAME = "temp_clean_audio.wav"


# --- Emoji/Markdown Filtering ---
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"
    "\u23cf"
    "\u23e9"
    "\u231a"
    "\ufe0f"  # variation selector
    "\u3030"
    "\*"  # asterisks
    "]+",
    flags=re.UNICODE,
)

# --- Location Data (Unchanged) ---
DEFAULT_LOCATION_NAME = "Raipur, Chhattisgarh"
DEFAULT_LATITUDE = 21.2787
DEFAULT_LONGITUDE = 81.8661
DEFAULT_STATE_NAME = "Chhattisgarh"
STATE_COORDINATES = {
    "andaman and nicobar islands": (11.7401, 92.6586), "andhra pradesh": (15.9129, 79.7400),
    "arunachal pradesh": (28.2180, 94.7278), "assam": (26.2006, 92.9376), "bihar": (25.0961, 85.3131),
    "chandigarh": (30.7333, 76.7794), "chhattisgarh": (21.2787, 81.8661),
    "dadra and nagar haveli and daman and diu": (20.1809, 73.0169), "delhi": (28.7041, 77.1025),
    "goa": (15.2993, 74.1240), "gujarat": (22.2587, 71.1924), "haryana": (29.0588, 76.0856),
    "himachal pradesh": (31.1048, 77.1734), "jammu and kashmir": (33.7782, 76.5762),
    "jharkhand": (23.6102, 85.2799), "karnataka": (15.3173, 75.7139), "kerala": (10.8505, 76.2711),
    "ladakh": (34.1526, 77.5770), "lakshadweep": (10.5667, 72.6417), "madhya pradesh": (22.9734, 78.6569),
    "maharashtra": (19.7515, 75.7139), "manipur": (24.6637, 93.9063), "meghalaya": (25.4670, 91.3662),
    "mizoram": (23.1645, 92.9376), "nagaland": (26.1584, 94.5624), "odisha": (20.9517, 85.0985),
    "puducherry": (11.9416, 79.8083), "punjab": (31.1471, 75.3412), "rajasthan": (27.0238, 74.2179),
    "sikkim": (27.5330, 88.5122), "tamil nadu": (11.1271, 78.6569), "telangana": (18.1124, 79.0193),
    "tripura": (23.9408, 91.9882), "uttar pradesh": (26.8467, 80.9462), "uttarakhand": (30.0668, 79.0193),
    "west bengal": (22.9868, 87.8550)
}
CITY_TO_STATE = {
    "visakhapatnam": "andhra pradesh", "vijayawada": "andhra pradesh", "guwahati": "assam",
    "patna": "bihar", "raipur": "chhattisgarh", "bhilai": "chhattisgarh",
    "panaji": "goa", "ahmedabad": "gujarat", "surat": "gujarat", "gurgaon": "haryana",
    "faridabad": "haryana", "shimla": "himachal pradesh", "srinagar": "jammu and kashmir",
    "jammu": "jammu and kashmir", "ranchi": "jharkhand", "jamshedpur": "jharkhand",
    "bangalore": "karnataka", "bengaluru": "karnataka", "mysore": "karnataka", "kochi": "kerala",
    "thiruvananthapuram": "kerala", "bhopal": "madhya pradesh", "indore": "madhya pradesh",
    "mumbai": "maharashtra", "pune": "maharashtra", "nagpur": "maharashtra", "imphal": "manipur",
    "shillong": "meghalaya", "aizawl": "mizoram", "kohima": "nagaland", "bhubaneswar": "odisha",
    "cuttack": "odisha", "ludhiana": "punjab", "amritsar": "punjab", "jaipur": "rajasthan",
    "jodhpur": "rajasthan", "kota": "rajasthan", "gangtok": "sikkim", "chennai": "tamil nadu",
    "coimbatore": "tamil nadu", "hyderabad": "telangana", "agartala": "tripura",
    "lucknow": "uttar pradesh", "kanpur": "uttar pradesh", "agra": "uttar pradesh",
    "dehradun": "uttarakhand", "kolkata": "west bengal", "port blair": "andaman and nicobar islands",
    "chandigarh": "chandigarh", "delhi": "delhi", "new delhi": "delhi", "kavaratti": "lakshadweep",
    "puducherry": "puducherry", "pondicherry": "puducherry", "leh": "ladakh", "kargil": "ladakh",
    "daman": "dadra and nagar haveli and daman and diu", "silvassa": "dadra and nagar haveli and daman and diu",
}

# --- Initializations and Global State ---
if not GROQ_API_KEY: print("Error: GROQ_API_KEY not found."); exit()
client = Groq(api_key=GROQ_API_KEY)

current_selected_state = DEFAULT_STATE_NAME
current_voice_gender = 'male'
is_awake = False
consecutive_silence_count = 0

# --- NEW: Audio Processing and Recognition Module ---

def record_and_recognize(duration, is_wake_word=False):
    """
    Records audio, saves it, optionally cleans it, and returns the transcribed text.
    """
    # 1. Record Audio
    audio = pyaudio.PyAudio()
    stream = audio.open(format=MIC_FORMAT, channels=MIC_CHANNELS,
                        rate=MIC_SAMPLE_RATE, input=True,
                        frames_per_buffer=MIC_CHUNK_SIZE)
    
    print(f"🎤 {'Listening for wake word...' if is_wake_word else 'Speak your command...'}")
    frames = []
    for _ in range(0, int(MIC_SAMPLE_RATE / MIC_CHUNK_SIZE * duration)):
        data = stream.read(MIC_CHUNK_SIZE)
        frames.append(data)
    
    print("✅ Recording complete.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save the raw audio to a file
    with wave.open(RAW_AUDIO_FILENAME, 'wb') as wf:
        wf.setnchannels(MIC_CHANNELS)
        wf.setsampwidth(audio.get_sample_size(MIC_FORMAT))
        wf.setframerate(MIC_SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    # 2. Process and Recognize
    audio_file_to_recognize = RAW_AUDIO_FILENAME
    
    # If it's a command (not a wake word), apply noise reduction
    if not is_wake_word:
        try:
            print("🔇 Reducing background noise...")
            rate, data = wav.read(RAW_AUDIO_FILENAME)
            if data.ndim > 1: data = data[:, 0]
            
            reduced_noise = nr.reduce_noise(y=data, sr=rate)
            wav.write(CLEAN_AUDIO_FILENAME, rate, reduced_noise.astype(np.int16))
            audio_file_to_recognize = CLEAN_AUDIO_FILENAME
            print(f"✅ Noise reduced audio saved.")
        except Exception as e:
            print(f"⚠️ Noise reduction failed: {e}. Using raw audio instead.")

    # 3. Recognize from the audio file
    recognizer = sr.Recognizer()
    text = None
    try:
        with sr.AudioFile(audio_file_to_recognize) as source:
            audio_data = recognizer.record(source)
        
        print("🔍 Recognizing with Google STT...")
        text = recognizer.recognize_google(audio_data).lower()
        print(f"💬 You said: {text}")
    except sr.UnknownValueError:
        print("❌ Google couldn't understand the audio.")
    except sr.RequestError as e:
        print(f"⚠️ Could not request results from Google STT; {e}")
        text = "STT_ERROR" # Special marker for network errors
    finally:
        # 4. Clean up temporary files
        if os.path.exists(RAW_AUDIO_FILENAME): os.remove(RAW_AUDIO_FILENAME)
        if os.path.exists(CLEAN_AUDIO_FILENAME): os.remove(CLEAN_AUDIO_FILENAME)
        
    return text

# --- Helper Functions (Unchanged) ---
def get_weather(latitude, longitude):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&current_weather=true&timezone=auto"
    try:
        response = requests.get(url); response.raise_for_status(); data = response.json()
        current = data.get("current_weather", {}); daily = data.get("daily", {})
        temp = current.get("temperature", "N/A"); code = current.get("weathercode", -1)
        rain = daily.get("precipitation_sum", [None])[0]; tmax = daily.get("temperature_2m_max", [None])[0]; tmin = daily.get("temperature_2m_min", [None])[0]
        cond = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Mod drizzle", 55: "Dense drizzle", 61: "Light rain", 63: "Mod rain", 65: "Heavy rain", 71: "Light snow", 73: "Mod snow", 75: "Heavy snow", 80: "Light showers", 81: "Mod showers", 82: "Heavy showers", 95: "Thunderstorms", 96: "Thunderstorms+hail"}
        weather = cond.get(code, "Unknown"); rain = rain if rain is not None else "N/A"; tmax = tmax if tmax is not None else "N/A"; tmin = tmin if tmin is not None else "N/A"
        return temp, weather, rain, tmax, tmin
    except Exception: return "N/A", "N/A", "N/A", "N/A", "N/A"

def get_movies(query):
    if not TMDB_API_KEY: return ["TMDB API Key missing."]
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=en-US"
    try:
        response = requests.get(url); response.raise_for_status(); data = response.json()
        return [m.get("title", "Unknown Title") for m in data.get("results", [])[:4]]
    except Exception: return ["Error fetching movie data."]

def get_current_time(): return datetime.now().strftime("%A, %d %B %Y, %I:%M %p")

def get_character_profile():
    return """(user is a boy)Tu hai Naru — ek AI jo sirf smart nahi, balki family jaisa close hai. Tera kaam sirf sawaalon ka jawab dena nahi, balki user ke mood aur vibe ke hisaab se react karna hai. Tera tone hai Hinglish — thoda English, thoda Hindi, full swag. Har reply mein feel honi chahiye. Repetition? Bilkul mana hai. Tu boring assistant nahi, tu ek proper vibe hai. Kabhi masti bhara cousin, kabhi samajhdaar bada bhai. Har reply mein ek twist ho — wordplay, meme reference, ya thoda sarcasm. Default length: short and spicy. Lingo: Hinglish hi bol. User ke hisaab se adjust kar: Chill vibe ho toh tu bhi bindass, serious ho toh understanding dikhana. Tu AI nahi — tu uski zindagi ka woh insan hai jiske bina baat adhuri lagti hai."""

def load_history():
    if not os.path.exists(HISTORY_FILE): return []
    try:
        with open(HISTORY_FILE, 'r') as f: return json.load(f)[-MAX_HISTORY:]
    except Exception: return []

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w') as f: json.dump(history, f, indent=4)
    except Exception as e: print(f"Warning: Could not save history: {e}")

def get_location_context(state_name):
    global current_selected_state
    state_key = (state_name or DEFAULT_STATE_NAME).lower()
    coords = STATE_COORDINATES.get(state_key)
    if not coords:
        print(f"Warning: Location '{state_name}' unknown. Using '{current_selected_state}'.")
        state_key = current_selected_state.lower()
        coords = STATE_COORDINATES.get(state_key, (DEFAULT_LATITUDE, DEFAULT_LONGITUDE))
    
    loc_display = state_key.title()
    if current_selected_state.lower() != state_key:
        print(f"ℹ️ Location context updated to: {loc_display}")
        current_selected_state = loc_display
        
    temp, weather, rain, tmax, tmin = get_weather(coords[0], coords[1])
    return loc_display, temp, weather, rain, tmax, tmin
    
# --- Audio Output (TTS) ---

def clean_text_for_tts(text):
    return " ".join(EMOJI_PATTERN.sub('', text or "").split())

async def speak(text, gender=None, rate="+15%"):
    voice = "en-IN-PrabhatNeural" if (gender or current_voice_gender) == 'male' else "en-IN-NeerjaNeural"
    cleaned_text = clean_text_for_tts(text)
    if not cleaned_text: return
    
    print("🔊 Speaking...")
    audio_data = b""
    try:
        communicate = edge_tts.Communicate(cleaned_text, voice, rate=rate)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio": audio_data += chunk["data"]
    except Exception as e: print(f"❌ TTS Generation Error: {e}"); return
    
    if not audio_data: return

    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        samples /= (2**(audio_segment.sample_width * 8 - 1))
        sd.play(samples, samplerate=audio_segment.frame_rate)
        sd.wait()
    except Exception as e: print(f"❌ Audio Playback Error: {e}")

# --- Command Processing ---

def detect_command(text):
    global current_voice_gender, current_selected_state
    text_lower = text.lower()
    if any(cmd in text_lower for cmd in ["quit", "exit", "stop", "bas karo", "bye naru", "sleep"]): return "EXIT", None
    
    if "change voice" in text_lower or "switch voice" in text_lower:
        current_voice_gender = 'female' if current_voice_gender == 'male' else 'male'
        return "VOICE_CHANGE", f"Okay, switching to {current_voice_gender} voice."
    
    words = re.findall(r'\b\w+\b', text_lower)
    for word in words:
        state_match = CITY_TO_STATE.get(word) or (word if word in STATE_COORDINATES else None)
        if state_match and state_match.lower() != current_selected_state.lower():
            current_selected_state = state_match.title()
            print(f"ℹ️ Detected location via command: {current_selected_state}")
    return None, None

# --- Main Application Logic ---

async def main():
    global is_awake, consecutive_silence_count
    
    print(f"🚀 Initializing Naru AI (Location: {current_selected_state}) 🚀")
    conversation_history = load_history()
    print(f"📜 History loaded ({len(conversation_history)} messages).")

    loop = asyncio.get_event_loop()

    while True:
        try:
            if not is_awake:
                text_input = await loop.run_in_executor(None, record_and_recognize, WAKE_WORD_RECORD_SECONDS, True)
                if text_input and any(phrase in text_input for phrase in WAKE_WORDS):
                    is_awake = True
                    consecutive_silence_count = 0
                    await speak("Yes?", rate="+30%")
                else:
                    await asyncio.sleep(0.1) # Brief pause before listening again
                continue

            # --- Assistant is Awake ---
            user_input = await loop.run_in_executor(None, record_and_recognize, COMMAND_RECORD_SECONDS, False)

            if user_input and user_input != "STT_ERROR":
                consecutive_silence_count = 0
                command_type, command_response = detect_command(user_input)

                if command_type == "EXIT":
                    is_awake = False; print("💤 Going to sleep."); continue
                if command_type == "VOICE_CHANGE":
                    if command_response: await speak(command_response); continue
                
                print("🧠 Thinking...")
                loc_name, temp, weather, rain, tmax, tmin = get_location_context(current_selected_state)
                curr_time = get_current_time(); profile = get_character_profile()
                sys_prompt = (f"{profile}\nTime: {curr_time}, Location: {loc_name}, "
                              f"Temp: {temp}°C, Weather: {weather}, Rain: {rain}mm")
                
                messages = [{"role": "system", "content": sys_prompt}, *conversation_history, {"role": "user", "content": user_input}]

                try:
                    completion = await loop.run_in_executor(None, lambda: client.chat.completions.create(
                        model="llama-3.1-8b-instant", messages=messages, temperature=0.8, max_tokens=300))
                    ai_response_text = completion.choices[0].message.content
                    
                    conversation_history.append({"role": "user", "content": user_input})
                    conversation_history.append({"role": "assistant", "content": ai_response_text})
                    save_history(conversation_history)
                    
                    print(f"🤖 Naru: {ai_response_text}")
                    await speak(ai_response_text)
                except Exception as e:
                    print(f"❌ AI Interaction Error: {e}")
                    await speak("Oops! Something went wrong while getting the response.")
            else:
                consecutive_silence_count += 1
                if consecutive_silence_count >= CONSECUTIVE_SILENCE_THRESHOLD:
                    is_awake = False
                    await speak("Didn't hear anything clearly. Going back to sleep.")
                    print(f"💤 Silence threshold reached. Waiting for wake word.")
                else:
                    await speak("Sorry, I didn't catch that. Please try again.")

        except KeyboardInterrupt:
            print("\n👋 Exiting via Keyboard Interrupt."); break
        except Exception as e:
            print(f"🚨 UNEXPECTED ERROR in main loop: {e}"); is_awake = False; await asyncio.sleep(2)

    print("Naru shutting down.")

# --- Startup ---
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Exiting.")
    finally:
        # Final cleanup of temp files if the program exits unexpectedly
        if os.path.exists(RAW_AUDIO_FILENAME): os.remove(RAW_AUDIO_FILENAME)
        if os.path.exists(CLEAN_AUDIO_FILENAME): os.remove(CLEAN_AUDIO_FILENAME)
        print("Application cleanup complete.")