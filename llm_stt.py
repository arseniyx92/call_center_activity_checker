from openai import OpenAI
from dotenv import load_dotenv
import os

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

# Получение токена из .env файла
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
  api_key = api_key
)

def preprocess_audio(file_path):
    """Предобработка аудио для улучшения качества распознавания"""
    if not AudioSegment:
        return file_path
    try:
        audio = AudioSegment.from_mp3(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1).normalize().low_pass_filter(8000)
        
        # Сохраняем в папку recordings с понятным именем
        recordings_dir = os.getenv('RECORDINGS_DIR', 'recordings')
        os.makedirs(recordings_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file = os.path.join(recordings_dir, f"{base_name}_processed.wav")
        
        audio.export(output_file, format="wav")
        print(f"✅ Обработанный аудио сохранен: {output_file}")
        return output_file
    except Exception as e:
        print(f"Audio preprocessing error: {e}")
        return file_path

def transcribe_mp3(file_path):
    processed_file = preprocess_audio(file_path)
    try:
        with open(processed_file, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
                language="ru",
                temperature=0.0,
                prompt="Ветеринарная клиника. Диалог оператора и клиента. Запись на прием, консультация."
            )
        return transcript
    except Exception as e:
        print(f"Error: {e}")
        return None