import hosted_pbx
import asyncio
import os
from tg_logger import set_application, Log_in_tg
from telegram.ext import Application
from dotenv import load_dotenv
from llm_stt import transcribe_mp3

# Load environment variables
load_dotenv()

recordings_dir = os.getenv('RECORDINGS_DIR')

def get_record_name(call):
    record_name = f"{call['start']}_{call['type']}_{call['id']}"
    return record_name

def setup_telegram():
    """Set up Telegram application"""
    BOT_TOKEN = os.getenv("LOG_BOT_TOKEN")
    if not BOT_TOKEN:
        raise ValueError("LOG_BOT_TOKEN environment variable not set!")
    
    application = Application.builder().token(BOT_TOKEN).build()
    set_application(application)
    print("✅ Telegram application set up successfully")
    return application

async def main():
    calls = hosted_pbx.get_call_history()
    if calls['error'] is not None:
        await Log_in_tg(f"❌ PBX API ERROR: {calls.error}")
        return

    await Log_in_tg(f"{calls['info'][-1]}")
    download_link = calls['info'][-1]['record']
    filename = get_record_name(calls['info'][-1])
    stt_result = None
    if hosted_pbx.download_recording(download_link, filename) == False:
        await Log_in_tg(f"❌ DOWNLOAD ERROR: Couldn't download file by this link {download_link}")
        return
    stt_result = transcribe_mp3(f'{recordings_dir}/{filename}')
    print(stt_result)
    await Log_in_tg(stt_result)


if __name__ == "__main__":
    setup_telegram()
    asyncio.run(main())