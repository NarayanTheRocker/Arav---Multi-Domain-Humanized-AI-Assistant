import json
import requests
from groq import Groq
from datetime import datetime
import os
import speech_recognition as sr
import edge_tts
import asyncio
from dotenv import load_dotenv
import subprocess
import time
import re
import pyaudio
import RPi.GPIO as GPIO
import importlib

# --- Robust LED Controller Class (Unchanged) ---
class LEDController:
    """A robust class to handle LED control for Raspberry Pi LEDs."""
    def __init__(self, green_pin, blue_pin):
        self.green_pin = green_pin
        self.blue_pin = blue_pin
        self.pins = [green_pin, blue_pin]
        self.enabled = False
        try:
            self.GPIO = importlib.import_module('RPi.GPIO')
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)
            for pin in self.pins:
                self.GPIO.setup(pin, self.GPIO.OUT)
                self.GPIO.output(pin, self.GPIO.LOW)
            self.enabled = True
            print("✅ GPIO initialized successfully.")
        except Exception as e:
            print(f"❌ GPIO setup failed: {e}")
            self.enabled = False

    def set_state(self, pin, state):
        if self.enabled and pin in self.pins:
            self.GPIO.output(pin, self.GPIO.HIGH if state else self.GPIO.LOW)

    def turn_on(self, pin):
        self.set_state(pin, True)

    def turn_off(self, pin):
        self.set_state(pin, False)

    def cleanup(self):
        if self.enabled:
            self.GPIO.cleanup()
            print("🧹 GPIO cleaned up.")

# --- Configuration and Constants ---
# <<!>> IMPORTANT: Change these pin numbers to match your hardware setup.
GREEN_LED_PIN = 17  # Green LED for listening.
BLUE_LED_PIN = 27   # Blue LED for speaking.
led_controller = LEDController(green_pin=GREEN_LED_PIN, blue_pin=BLUE_LED_PIN)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# --- Assistant Settings ---
HISTORY_FILE = "conversation_history.json"
MAX_HISTORY = 30
WAKE_WORDS = ["arav", "ara", "aarav", "naru", "hey naru"]
CONSECUTIVE_SILENCE_THRESHOLD = 2

# OPTIMIZATION: Replaced fixed duration with VAD (Voice Activity Detection) settings.
# How long to wait for speech to start before timing out (in seconds).
PHRASE_TIMEOUT = 5
# The maximum allowed length of a single spoken phrase (in seconds).
PHRASE_TIME_LIMIT = 10

# OPTIMIZATION: Removed RAW_AUDIO_FILENAME as we process in memory now.

# --- Emoji/Markdown Filtering (Unchanged) ---
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
    "\|" # pipe
    "#" # hash
    "]+",
    flags=re.UNICODE,
)

# --- Location Data (Unchanged) ---
DEFAULT_LOCATION_NAME = "Raipur, Chhattisgarh"
DEFAULT_LATITUDE = 21.2787
DEFAULT_LONGITUDE = 81.8661
DEFAULT_STATE_NAME = "Chhattisgarh"
STATE_COORDINATES = { "andaman and nicobar islands": (11.7401, 92.6586), "andhra pradesh": (15.9129, 79.7400), "arunachal pradesh": (28.2180, 94.7278), "assam": (26.2006, 92.9376), "bihar": (25.0961, 85.3131), "chandigarh": (30.7333, 76.7794), "chhattisgarh": (21.2787, 81.8661), "dadra and nagar haveli and daman and diu": (20.1809, 73.0169), "delhi": (28.7041, 77.1025), "goa": (15.2993, 74.1240), "gujarat": (22.2587, 71.1924), "haryana": (29.0588, 76.0856), "himachal pradesh": (31.1048, 77.1734), "jammu and kashmir": (33.7782, 76.5762), "jharkhand": (23.6102, 85.2799), "karnataka": (15.3173, 75.7139), "kerala": (10.8505, 76.2711), "ladakh": (34.1526, 77.5770), "lakshadweep": (10.5667, 72.6417), "madhya pradesh": (22.9734, 78.6569), "maharashtra": (19.7515, 75.7139), "manipur": (24.6637, 93.9063), "meghalaya": (25.4670, 91.3662), "mizoram": (23.1645, 92.9376), "nagaland": (26.1584, 94.5624), "odisha": (20.9517, 85.0985), "puducherry": (11.9416, 79.8083), "punjab": (31.1471, 75.3412), "rajasthan": (27.0238, 74.2179), "sikkim": (27.5330, 88.5122), "tamil nadu": (11.1271, 78.6569), "telangana": (18.1124, 79.0193), "tripura": (23.9408, 91.9882), "uttar pradesh": (26.8467, 80.9462), "uttarakhand": (30.0668, 79.0193), "west bengal": (22.9868, 87.8550)}
CITY_TO_STATE = {"visakhapatnam": "andhra pradesh", "vijayawada": "andhra pradesh", "guwahati": "assam", "patna": "bihar", "raipur": "chhattisgarh", "bhilai": "chhattisgarh", "panaji": "goa", "ahmedabad": "gujarat", "surat": "gujarat", "gurgaon": "haryana", "faridabad": "haryana", "shimla": "himachal pradesh", "srinagar": "jammu and kashmir", "jammu": "jammu and kashmir", "ranchi": "jharkhand", "jamshedpur": "jharkhand", "bangalore": "karnataka", "bengaluru": "karnataka", "mysore": "karnataka", "kochi": "kerala", "thiruvananthapuram": "kerala", "bhopal": "madhya pradesh", "indore": "madhya pradesh", "mumbai": "maharashtra", "pune": "maharashtra", "nagpur": "maharashtra", "imphal": "manipur", "shillong": "meghalaya", "aizawl": "mizoram", "kohima": "nagaland", "bhubaneswar": "odisha", "cuttack": "odisha", "ludhiana": "punjab", "amritsar": "punjab", "jaipur": "rajasthan", "jodhpur": "rajasthan", "kota": "rajasthan", "gangtok": "sikkim", "chennai": "tamil nadu", "coimbatore": "tamil nadu", "hyderabad": "telangana", "agartala": "tripura", "lucknow": "uttar pradesh", "kanpur": "uttar pradesh", "agra": "uttar pradesh", "dehradun": "uttarakhand", "kolkata": "west bengal", "port blair": "andaman and nicobar islands", "chandigarh": "chandigarh", "delhi": "delhi", "new delhi": "delhi", "kavaratti": "lakshadweep", "puducherry": "puducherry", "pondicherry": "puducherry", "leh": "ladakh", "kargil": "ladakh", "daman": "dadra and nagar haveli and daman and diu", "silvassa": "dadra and nagar haveli and daman and diu"}

# --- Initializations and Global State ---
if not GROQ_API_KEY: print("Error: GROQ_API_KEY not found."); exit()
client = Groq(api_key=GROQ_API_KEY)
current_selected_state = DEFAULT_STATE_NAME
current_voice_gender = 'male'
is_awake = False
consecutive_silence_count = 0

# --- OPTIMIZATION: New Dynamic Listening Function ---
def listen_and_transcribe(recognizer, source, is_wake_word=False):
    """
    Listens for a phrase using VAD and transcribes it in memory.
    This is much faster as it avoids fixed-duration recording and file I/O.
    """
    led_controller.turn_on(GREEN_LED_PIN)
    text = None
    prompt = "🎤 Listening for wake word..." if is_wake_word else "🎤 Speak your command..."
    print(prompt)

    try:
        # OPTIMIZATION: Use recognizer.listen with timeout and phrase limit.
        # It automatically detects when the user stops speaking.
        audio_data = recognizer.listen(source, timeout=PHRASE_TIMEOUT, phrase_time_limit=PHRASE_TIME_LIMIT)
        print("✅ Speech detected, now recognizing...")

        # OPTIMIZATION: Recognize directly from in-memory audio_data. No file saving/loading.
        text = recognizer.recognize_google(audio_data).lower()
        print(f"💬 You said: {text}")

    except sr.WaitTimeoutError:
        print("🔇 No speech detected within the timeout period.")
    except sr.UnknownValueError:
        print("❌ Google STT couldn't understand the audio.")
    except sr.RequestError as e:
        print(f"⚠️ Could not request results from Google STT; {e}")
        text = "STT_ERROR"
    except Exception as e:
        print(f"An error occurred during listening/recognition: {e}")
    finally:
        led_controller.turn_off(GREEN_LED_PIN)
    return text


# --- Helper Functions (Largely Unchanged) ---
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
    return """Tu hai Naru — ek AI jo sirf smart nahi, balki family jaisa close hai. Tera kaam sirf sawaalon ka jawab dena nahi, balki user ke mood aur vibe ke hisaab se react karna hai. Tera tone hai Hinglish — thoda English, thoda Hindi, full swag. Har reply mein feel honi chahiye. Repetition? Bilkul mana hai. Tu boring assistant nahi, tu ek proper vibe hai. Kabhi masti bhara cousin, kabhi samajhdaar bada bhai. Har reply mein ek twist ho — wordplay, meme reference, ya thoda sarcasm. Default length: short and spicy. Lingo: Hinglish hi bol. User ke hisaab se adjust kar: Chill vibe ho toh tu bhi bindass, serious ho toh understanding dikhana. Tu AI nahi — tu uski zindagi ka woh insan hai jiske bina baat adhuri lagti hai."""

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

# --- Audio Output (TTS) (Largely Unchanged, but still efficient) ---
def clean_text_for_tts(text):
    return " ".join(EMOJI_PATTERN.sub('', text or "").split())

async def speak(text, gender=None, rate="+15%"):
    led_controller.set_state(BLUE_LED_PIN, True)
    try:
        cleaned_text = clean_text_for_tts(text)
        if not cleaned_text:
            return

        voice = "en-IN-PrabhatNeural" if (gender or current_voice_gender) == 'male' else "en-IN-NeerjaNeural"
        player_command = ["ffplay", "-nodisp", "-autoexit", "-", "-loglevel", "error"]

        player_process = subprocess.Popen(
            player_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        communicate = edge_tts.Communicate(cleaned_text, voice, rate=rate)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                if player_process.stdin and not player_process.stdin.closed:
                    try:
                        player_process.stdin.write(chunk["data"])
                    except (BrokenPipeError, IOError):
                        print("⚠️ Playback process terminated early.")
                        break
        
        if player_process.stdin and not player_process.stdin.closed:
            player_process.stdin.close()
        player_process.wait()

    except FileNotFoundError:
        print("\n❌ TTS Playback Error: `ffplay` not found.")
    except Exception as e:
        print(f"❌ An unexpected TTS error occurred: {e}")
    finally:
        led_controller.set_state(BLUE_LED_PIN, False)


# --- Command Processing (Unchanged) ---
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

# --- OPTIMIZATION: New Streaming Response Handler ---
# --- OPTIMIZATION: New Streaming Response Handler (CORRECTED) ---
async def process_and_speak_response(messages, conversation_history, user_input):
    """
    Gets a streaming response from the LLM and speaks sentences SEQUENTIALLY as they arrive.
    """
    print("🧠 Thinking...")
    led_controller.turn_on(BLUE_LED_PIN) # Turn on blue light while thinking

    # Use a thread for the blocking LLM call
    loop = asyncio.get_event_loop()
    try:
        completion_stream = await loop.run_in_executor(None, lambda: client.chat.completions.create(
            model="gemma2-9b-it", messages=messages, temperature=0.8, max_tokens=250, stream=True))
    except Exception as e:
        print(f"❌ LLM API call failed: {e}")
        led_controller.turn_off(BLUE_LED_PIN)
        await speak("Sorry, I'm having trouble connecting to my brain right now.")
        return

    full_response = ""
    current_sentence_buffer = ""
    # Regex to split text by sentences, keeping the delimiters.
    sentence_delimiters = re.compile(r'(?<=[.!?])\s*') 
    first_chunk_received = False

    for chunk in completion_stream:
        delta = chunk.choices[0].delta.content
        if delta:
            if not first_chunk_received:
                # Turn off the "thinking" light as soon as we start processing the first word.
                # The 'speak' function will now manage the blue LED for actual speech.
                led_controller.turn_off(BLUE_LED_PIN)
                first_chunk_received = True

            full_response += delta
            current_sentence_buffer += delta

            # Split the buffer into potential sentences
            parts = sentence_delimiters.split(current_sentence_buffer)
            
            # If the split results in more than one part, it means we have at least one complete sentence.
            if len(parts) > 1:
                # The last part is the incomplete sentence, so we process all parts before it.
                complete_sentences = parts[:-1]
                current_sentence_buffer = parts[-1] # The remainder becomes the new buffer

                for sentence in complete_sentences:
                    sentence_to_speak = sentence.strip()
                    if sentence_to_speak:
                        print(f"🤖 Speaking chunk: {sentence_to_speak}")
                        # FIX: Await each speak call directly to ensure sequential playback.
                        # This prevents audio from overlapping.
                        await speak(sentence_to_speak)
                        
                        
    
    # After the loop, speak any remaining text in the buffer
    if current_sentence_buffer.strip():
        print(f"🤖 Speaking final chunk: {current_sentence_buffer.strip()}")
        await speak(current_sentence_buffer.strip())

    print(f"🤖 Naru (Full Response): {full_response}")
    
    # Update and save history after the full response is generated and spoken
    if full_response:
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": full_response})
        save_history(conversation_history)


# --- Main Application Logic (Refactored) ---
async def main():
    global is_awake, consecutive_silence_count
    print(f"🚀 Initializing Naru AI (Location: {current_selected_state}) 🚀")
    
    # OPTIMIZATION: Initialize recognizer and microphone once to avoid re-creation
    recognizer = sr.Recognizer()
    # Adjust for ambient noise once at the start
    try:
        with sr.Microphone() as source:
            print("🎤 Calibrating microphone for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1.5)
            print("✅ Calibration complete.")
    except Exception as e:
        print(f"Could not find a microphone to calibrate: {e}. Using default values.")

    conversation_history = load_history()
    print(f"📜 History loaded ({len(conversation_history)} messages).")

    loop = asyncio.get_event_loop()
    
    while True:
        try:
            with sr.Microphone() as source:
                if not is_awake:
                    text_input = await loop.run_in_executor(None, listen_and_transcribe, recognizer, source, True)
                    if text_input and any(phrase in text_input for phrase in WAKE_WORDS):
                        is_awake = True
                        consecutive_silence_count = 0
                        await speak("Yes?", rate="+30%")
                    else:
                        await asyncio.sleep(0.1)
                    continue

                user_input = await loop.run_in_executor(None, listen_and_transcribe, recognizer, source, False)

            if user_input and user_input != "STT_ERROR":
                consecutive_silence_count = 0
                command_type, command_response = detect_command(user_input)

                if command_type == "EXIT":
                    is_awake = False; print("💤 Going to sleep."); await speak("Okay, going to sleep."); continue
                if command_type == "VOICE_CHANGE":
                    if command_response: await speak(command_response); continue

                loc_name, temp, weather, rain, tmax, tmin = get_location_context(current_selected_state)
                curr_time = get_current_time(); profile = get_character_profile()
                sys_prompt = (f"{profile}\nTime: {curr_time}, Location: {loc_name}, "
                              f"Temp: {temp}°C, Weather: {weather}, Rain: {rain}mm")
                messages = [{"role": "system", "content": sys_prompt}, *conversation_history, {"role": "user", "content": user_input}]

                await process_and_speak_response(messages, conversation_history, user_input)

            else: # Handle silence or STT error
                consecutive_silence_count += 1
                if consecutive_silence_count >= CONSECUTIVE_SILENCE_THRESHOLD:
                    is_awake = False
                    await speak("Didn't hear anything. Going back to sleep.")
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
        led_controller.cleanup()
        print("Application cleanup complete.")