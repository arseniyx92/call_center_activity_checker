#!/bin/bash

# Start Telegram bot in background
python3 tg_bot_core.py &

# Save the background process PID
BOT_PID=$!

# Wait a moment for bot to initialize
sleep 3

# Start main script (in foreground)
python3 main_script.py

# When main script exits, also kill the bot
kill $BOT_PID