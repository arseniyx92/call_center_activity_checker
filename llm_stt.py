from openai import OpenAI
from dotenv import load_dotenv
import os

# Получение токена из .env файла
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
  api_key = api_key
)

def transcribe_mp3(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"  # Can be "json", "text", "srt", "verbose_json", "vtt"
            )
        return transcript
    except Exception as e:
        print(f"Error: {e}")
        return None