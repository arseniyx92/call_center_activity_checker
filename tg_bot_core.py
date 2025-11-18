from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
from dotenv import load_dotenv
from tg_logger import set_application, Log_in_tg  # Import from our module

# Load environment variables
load_dotenv()

BOT_TOKEN = os.getenv("LOG_BOT_TOKEN")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Example command that uses the logger"""
    await update.message.reply_text("Bot started!")
    
    # Use the logger from another module
    await Log_in_tg(f"User {update.effective_user.id} started the bot")

async def test_log_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command to test the logger"""
    success = await Log_in_tg("Test message from bot command!")
    if success:
        await update.message.reply_text("✅ Log message sent to admins")
    else:
        await update.message.reply_text("❌ Failed to send log message")

def main():
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Set the application instance for the logger
    set_application(application)
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("test_log", test_log_command))
    
    # Start the bot
    application.run_polling(poll_interval=3)

if __name__ == "__main__":
    main()