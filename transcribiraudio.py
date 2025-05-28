from pydub import AudioSegment
import speech_recognition as sr

# Cargar el archivo de audio
audio = AudioSegment.from_ogg("C:\\Users\\34644\\Downloads\\WhatsApp.ogg")

# Convertir el audio a formato .wav
audio.export("converted_audio.wav", format="wav")

# Inicializar el reconocedor
recognizer = sr.Recognizer()

# Abrir y transcribir el archivo de audio convertido
with sr.AudioFile("converted_audio.wav") as source:
    audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data, language="es-ES")  # Suponiendo que el audio está en español
    print(text)
