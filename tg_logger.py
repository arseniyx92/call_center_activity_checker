import os
from telegram.error import TelegramError
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get admin users
admin_users_str = os.getenv("ADMIN_TELEGRAM_USERS", "").strip()
ADMIN_USERS = [int(user_id.strip()) for user_id in admin_users_str.split(",")] if admin_users_str else []

logger = logging.getLogger(__name__)

# Store the application instance globally
_application = None

def set_application(app):
    """Set the application instance for sending messages"""
    global _application
    _application = app

async def Log_in_tg(
    message: str,
    parse_mode: str = None,
    reply_markup = None
) -> bool:
    if not _application:
        logger.error("Application not set! Call set_application() first.")
        return False
    
    if not ADMIN_USERS:
        logger.error("No admin users configured!")
        return False
    
    success_count = 0
    for user_id in ADMIN_USERS:
        try:
            await _application.bot.send_message(
                chat_id=user_id,
                text=message,
                parse_mode=parse_mode,
                reply_markup=reply_markup
            )
            success_count += 1
            logger.info(f"Message sent to admin {user_id}")
            
        except TelegramError as e:
            logger.error(f"Failed to send message to admin {user_id}: {e}")
    
    return success_count > 0