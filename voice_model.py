import speech_recognition as sr
import wave
import pyaudio

def record_audio(filename="voice_input.wav", record_seconds=5, sample_rate=16000):
    chunk = 1024
    fmt = pyaudio.paInt16
    channels = 1

    audio = pyaudio.PyAudio()
    stream = audio.open(format=fmt,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("🎤 Recording... Speak now!")
    frames = []

    for _ in range(0, int(sample_rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("✅ Done Recording.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recording
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(fmt))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

    print(f"💾 Audio saved to {filename}")
    return filename


def recognize_from_audio(filename):
    recognizer = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio_data = recognizer.record(source)

    try:
        print("🔍 Recognizing with Google STT...")
        text = recognizer.recognize_google(audio_data)
        print("📝 You said:", text)
    except sr.UnknownValueError:
        print("❌ Google couldn't understand the audio.")
    except sr.RequestError as e:
        print(f"⚠️ Could not request results from Google STT; {e}")

if __name__ == "__main__":
    wav_file = record_audio(record_seconds=6)  # You can change the duration
    recognize_from_audio(wav_file)
