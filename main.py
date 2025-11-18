import hosted_pbx
import asyncio
import os
from tg_logger import set_application, Log_in_tg
from telegram.ext import Application
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
        await Log_in_tg(f"❌ PBX_API_ERROR: {calls.error}")

    await Log_in_tg(f"{calls['info'][-1]}")
    hosted_pbx.download_recording(calls['info'][-1]['record'], 'recording.mp3')

if __name__ == "__main__":
    setup_telegram()
    asyncio.run(main())