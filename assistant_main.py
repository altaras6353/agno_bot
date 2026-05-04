import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
import pytz
from dateutil import parser as dateutil_parser
import psycopg2
from psycopg2.extras import RealDictCursor
from apscheduler.schedulers.background import BackgroundScheduler
from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.db.postgres import PostgresDb
from agno.tools.google.calendar import GoogleCalendarTools
import telebot
from dotenv import load_dotenv

import soundfile as sf
import speech_recognition as sr
import io
from agno.media import Image

from flask import Flask, request
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/postgres")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PORT = int(os.getenv("PORT", 5000))

if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is missing!")
    exit(1)

# Write credentials.json from environment variable if provided (for Railway deployment)
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")
if GOOGLE_CREDENTIALS_JSON and not os.path.exists("credentials.json"):
    with open("credentials.json", "w") as f:
        f.write(GOOGLE_CREDENTIALS_JSON)
    logger.info("credentials.json written from environment variable.")

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# ==========================================
# Google OAuth Configuration
# ==========================================
SCOPES = ['https://www.googleapis.com/auth/calendar']
CLIENT_SECRETS_FILE = "credentials.json"
# Dynamic redirect URI: use Railway public domain if available, else localhost
_railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
REDIRECT_URI = f"https://{_railway_domain}/oauth2callback" if _railway_domain else f"http://localhost:{PORT}/oauth2callback"

# Store PKCE code verifiers keyed by user_id (state)
_pending_code_verifiers = {}

def get_auth_url(user_id: str) -> str:
    """Generates the Google OAuth authorization URL for a given user."""
    if not os.path.exists(CLIENT_SECRETS_FILE):
        return "Error: credentials.json missing. The administrator must provide Google OAuth credentials."
    
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    # Use 'state' to pass the user_id through the OAuth flow
    auth_url, _ = flow.authorization_url(prompt='consent', state=user_id, access_type='offline')
    # Store the PKCE code_verifier so it can be used in the callback
    _pending_code_verifiers[user_id] = flow.code_verifier
    return auth_url

# ==========================================
# Database Connections & Initialization
# ==========================================
def get_db_connection():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)

def transcribe_ogg(ogg_path: str) -> str:
    data, samplerate = sf.read(ogg_path)
    wav_io = io.BytesIO()
    sf.write(wav_io, data, samplerate, format='WAV', subtype='PCM_16')
    wav_io.seek(0)
    
    r = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio = r.record(source)
    return r.recognize_google(audio, language='he-IL')

def init_db():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_facts (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    fact_text TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id SERIAL PRIMARY KEY,
                    user_id VARCHAR(255) NOT NULL,
                    message TEXT NOT NULL,
                    remind_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # NEW: Table for storing Google OAuth tokens
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_tokens (
                    user_id VARCHAR(255) PRIMARY KEY,
                    token_json TEXT NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)
        conn.commit()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing DB: {e}")
        conn.rollback()
    finally:
        conn.close()

# ==========================================
# Custom Agent Tools
# ==========================================
def add_fact(user_id: str, fact_text: str) -> str:
    """Saves a long-term fact about the user in the database."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO user_facts (user_id, fact_text) VALUES (%s, %s)", (user_id, fact_text))
        conn.commit()
        return f"Successfully remembered: {fact_text}"
    except Exception as e:
        conn.rollback()
        return f"Error saving fact: {e}"
    finally:
        conn.close()

def get_facts(user_id: str) -> str:
    """Retrieves all long-term facts stored for this user."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT fact_text FROM user_facts WHERE user_id = %s", (user_id,))
            rows = cur.fetchall()
            if not rows:
                return "No facts found for this user."
            return "\n".join(row['fact_text'] for row in rows)
    except Exception as e:
        return f"Error retrieving facts: {e}"
    finally:
        conn.close()

def parse_iso_time(iso_str: str) -> datetime:
    """Parse an ISO 8601 datetime string."""
    try:
        return dateutil_parser.isoparse(iso_str)
    except Exception:
        return None

def schedule_reminder(user_id: str, message: str, remind_at_iso: str) -> str:
    """Schedules a future reminder for the user (ISO 8601 format). Converts UTC times to local timezone."""
    local_tz = pytz.timezone("Asia/Jerusalem")
    
    conn = get_db_connection()
    try:
        # Parse the ISO string and convert to local time
        parsed_dt = parse_iso_time(remind_at_iso)
        if parsed_dt is None:
            return f"Error: Invalid datetime format: {remind_at_iso}"
        
        # Ensure the datetime is in local timezone
        if parsed_dt.tzinfo is None:
            # Naive datetime - assume it's local time
            local_dt = local_tz.localize(parsed_dt)
        else:
            # Aware datetime - convert to local time
            local_dt = parsed_dt.astimezone(local_tz)
        
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO reminders (user_id, message, remind_at) VALUES (%s, %s, %s)",
                (user_id, message, local_dt)
            )
        conn.commit()
        return f"Reminder scheduled for {local_dt.strftime('%d/%m/%Y %H:%M')}."
    except Exception as e:
        conn.rollback()
        return f"Error scheduling reminder: {e}"
    finally:
        conn.close()

def get_user_credentials(user_id: str):
    """Retrieves the Google Credentials object from the DB for a specific user."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT token_json FROM user_tokens WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
            if row:
                creds_data = json.loads(row['token_json'])
                return Credentials.from_authorized_user_info(creds_data, SCOPES)
    except Exception as e:
        logger.error(f"Error reading credentials for {user_id}: {e}")
    finally:
        conn.close()
    return None

# ==========================================
# Agent Setup
# ==========================================
def create_agent(user_id: str) -> Agent:
    db = PostgresDb(db_url=DB_URL, session_table="agent_sessions")
    
    def add_user_fact(fact_text: str) -> str:
        """Saves a long-term fact about the user in the database."""
        return add_fact(user_id, fact_text)

    def get_user_facts() -> str:
        """Retrieves all long-term facts stored for this user."""
        return get_facts(user_id)

    def schedule_user_reminder(message: str, remind_at_iso: str) -> str:
        """Schedules a future reminder for the user (ISO 8601 format)."""
        return schedule_reminder(user_id, message, remind_at_iso)

    def get_user_reminders() -> str:
        """Retrieves all pending reminders for the user. Use this to find the ID of a reminder to update or cancel."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id, message, remind_at FROM reminders WHERE user_id = %s AND status = 'pending'", (user_id,))
                rows = cur.fetchall()
                if not rows:
                    return "No pending reminders."
                return "\n".join(f"ID: {r['id']} | Time: {r['remind_at']} | Msg: {r['message']}" for r in rows)
        except Exception as e:
            return f"Error: {e}"
        finally:
            conn.close()

    def update_user_reminder(reminder_id: int, new_message: str = None, new_remind_at_iso: str = None) -> str:
        """Updates an existing reminder. Find the ID using get_user_reminders first."""
        if not new_message and not new_remind_at_iso:
            return "You must provide either new_message or new_remind_at_iso to update."

        local_tz = pytz.timezone("Asia/Jerusalem")
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM reminders WHERE id = %s AND user_id = %s AND status = 'pending'", (reminder_id, user_id))
                if not cur.fetchone():
                    return f"Reminder ID {reminder_id} not found or not pending."

                new_remind_at = None
                if new_remind_at_iso:
                    parsed_dt = parse_iso_time(new_remind_at_iso)
                    if parsed_dt is None:
                        return f"Error: Invalid datetime format: {new_remind_at_iso}"
                    if parsed_dt.tzinfo is None:
                        new_remind_at = local_tz.localize(parsed_dt)
                    else:
                        new_remind_at = parsed_dt.astimezone(local_tz)

                if new_message and new_remind_at:
                    cur.execute("UPDATE reminders SET message = %s, remind_at = %s WHERE id = %s", (new_message, new_remind_at, reminder_id))
                elif new_message:
                    cur.execute("UPDATE reminders SET message = %s WHERE id = %s", (new_message, reminder_id))
                elif new_remind_at:
                    cur.execute("UPDATE reminders SET remind_at = %s WHERE id = %s", (new_remind_at, reminder_id))
            conn.commit()
            return f"Successfully updated reminder ID {reminder_id}."
        except Exception as e:
            conn.rollback()
            return f"Error updating reminder: {e}"
        finally:
            conn.close()

    def cancel_user_reminder(reminder_id: int) -> str:
        """Cancels a pending reminder. Find the ID using get_user_reminders first."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE reminders SET status = 'cancelled' WHERE id = %s AND user_id = %s AND status = 'pending'", (reminder_id, user_id))
                if cur.rowcount == 0:
                    return f"Reminder ID {reminder_id} not found or not pending."
            conn.commit()
            return f"Successfully cancelled reminder ID {reminder_id}."
        except Exception as e:
            conn.rollback()
            return f"Error cancelling reminder: {e}"
        finally:
            conn.close()

    tools = [add_user_fact, get_user_facts, schedule_user_reminder, get_user_reminders, update_user_reminder, cancel_user_reminder]
    
    # Check if user has authenticated with Google Calendar
    creds = get_user_credentials(user_id)
    if creds:
        # User is authenticated, inject real Calendar Tools
        tools.append(GoogleCalendarTools(creds=creds))
    else:
        # User is NOT authenticated, inject a tool to request auth
        def request_calendar_auth() -> str:
            """Call this when the user wants to interact with Google Calendar but isn't authenticated yet."""
            link = get_auth_url(user_id)
            return f"Tell the user they must authenticate first by clicking this link: {link}"
        tools.append(request_calendar_auth)

    now = datetime.now().astimezone()
    current_time_str = now.isoformat()
    current_weekday = now.strftime("%A")
    
    from datetime import timedelta
    calendar_ref = "\n        ".join([(now + timedelta(days=i)).strftime("%A, %Y-%m-%d") for i in range(14)])

    return Agent(
        model=OpenRouter(id="openai/gpt-4o-mini", api_key=OPENROUTER_API_KEY),
        db=db,
        tools=tools,
        description=f"""You are a highly capable personal assistant.
        The current date and time is {current_time_str} ({current_weekday}).
        Here is a calendar reference for the next 14 days to help you calculate dates accurately:
        {calendar_ref}

        You manage memory, Google Calendar events, and Telegram Reminders.
        IMPORTANT DISTINCTION:
        1. "Reminders" (תזכורות) are in-app Telegram notifications sent at a specific time. Use schedule_user_reminder, get_user_reminders, update_user_reminder, and cancel_user_reminder.
        2. "Calendar Events" (פגישות / יומן / משמרות) are saved in Google Calendar. Use the calendar tools (create_event, search_events, update_event, delete_event).

        If the user wants to update or delete a Calendar Event, ALWAYS use search_events or list_events first to find the correct Event ID, and then use update_event or delete_event.
        Always check the user's facts using get_user_facts before making assumptions.
        Respond in Hebrew.""",
        user_id=str(user_id),
        session_id=str(user_id),
        read_chat_history=True,
    )

# ==========================================
# Background Scheduler (Proactive Reminders)
# ==========================================
def check_and_send_reminders():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, user_id, message FROM reminders 
                WHERE status = 'pending' AND remind_at <= CURRENT_TIMESTAMP
            """)
            reminders = cur.fetchall()
            
            for r in reminders:
                logger.info(f"Sending reminder {r['id']} to {r['user_id']}")
                try:
                    markup = InlineKeyboardMarkup()
                    markup.add(InlineKeyboardButton("✅ בוצע", callback_data=f"rem_done_{r['id']}"))
                    markup.add(InlineKeyboardButton("⏳ עוד 15 דקות", callback_data=f"rem_snooze_15_{r['id']}"))
                    markup.add(InlineKeyboardButton("⏳ עוד שעה", callback_data=f"rem_snooze_60_{r['id']}"))
                    markup.add(InlineKeyboardButton("📅 מחר", callback_data=f"rem_snooze_1440_{r['id']}"))
                    
                    bot.send_message(chat_id=r['user_id'], text=f"🔔 *תזכורת:*\n{r['message']}", parse_mode="Markdown", reply_markup=markup)
                    cur.execute("UPDATE reminders SET status = 'sent' WHERE id = %s", (r['id'],))
                except Exception as e:
                    logger.error(f"Failed to send message: {e}")
                    cur.execute("UPDATE reminders SET status = 'failed' WHERE id = %s", (r['id'],))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error processing reminders: {e}")
    finally:
        conn.close()

def start_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(check_and_send_reminders, 'interval', minutes=1)
    scheduler.start()
    logger.info("Scheduler started.")

# ==========================================
# Flask App for OAuth Redirect
# ==========================================
app = Flask(__name__)

@app.route('/oauth2callback')
def oauth2callback():
    state = request.args.get('state') # This contains the Telegram user_id
    code = request.args.get('code')
    
    if not state or not code:
        return "Invalid request.", 400
        
    try:
        flow = Flow.from_client_secrets_file(
            CLIENT_SECRETS_FILE,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        # Restore the PKCE code_verifier from the original auth request
        code_verifier = _pending_code_verifiers.pop(state, None)
        flow.fetch_token(code=code, code_verifier=code_verifier)
        creds = flow.credentials
        
        # Save credentials to the database
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_tokens (user_id, token_json) 
                VALUES (%s, %s)
                ON CONFLICT (user_id) 
                DO UPDATE SET token_json = EXCLUDED.token_json, updated_at = CURRENT_TIMESTAMP
            """, (state, creds.to_json()))
        conn.commit()
        conn.close()
        
        # Notify the user via Telegram
        bot.send_message(chat_id=state, text="✅ התחברות ליומן גוגל בוצעה בהצלחה! כעת תוכל לקבוע פגישות דרכי.")
        return "Authentication successful! You can safely close this window and return to Telegram.", 200
    except Exception as e:
        logger.error(f"OAuth error: {e}")
        return f"Authentication failed: {e}", 500

def start_flask():
    # Run Flask on Railway's injected PORT, or 5000 locally
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

# ==========================================
# Telegram Bot Handlers
# ==========================================

def get_main_keyboard():
    markup = ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add(KeyboardButton("📅 חיבורים / Auths"), KeyboardButton("⏰ התזכורות שלי"))
    return markup

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "שלום! אני העוזר האישי שלך. מוזמן לדבר איתי חופשי.\n\nהשתמש בכפתור למטה כדי לנהל את החיבורים החיצוניים שלך או לצפות בתזכורות.", reply_markup=get_main_keyboard())

@bot.message_handler(func=lambda message: message.text == "⏰ התזכורות שלי")
def show_my_reminders(message):
    user_id = str(message.chat.id)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, message, remind_at FROM reminders WHERE user_id = %s AND status = 'pending' ORDER BY remind_at ASC", (user_id,))
            rows = cur.fetchall()
            if not rows:
                bot.reply_to(message, "אין לך תזכורות עתידיות.", reply_markup=get_main_keyboard())
                return
            
            response = "*התזכורות שלך:*\n\n"
            for r in rows:
                dt = r['remind_at']
                time_str = dt.strftime("%d/%m/%Y %H:%M")
                response += f"🔹 *{time_str}*\n{r['message']}\n\n"
                
            bot.reply_to(message, response, parse_mode="Markdown", reply_markup=get_main_keyboard())
    except Exception as e:
        logger.error(f"Error fetching reminders for {user_id}: {e}")
        bot.reply_to(message, "אירעה שגיאה בשליפת התזכורות.", reply_markup=get_main_keyboard())
    finally:
        conn.close()

@bot.message_handler(func=lambda message: message.text == "📅 חיבורים / Auths")
def manage_auths(message):
    user_id = str(message.chat.id)
    creds = get_user_credentials(user_id)
    
    markup = InlineKeyboardMarkup()
    if creds:
        status = "✅ יומן גוגל: מחובר"
        markup.add(InlineKeyboardButton("🔗 התחברות מחדש", url=get_auth_url(user_id)))
    else:
        status = "❌ יומן גוגל: לא מחובר"
        markup.add(InlineKeyboardButton("🔗 לחץ כאן להתחברות", url=get_auth_url(user_id)))
        
    bot.reply_to(message, f"*ניהול חיבורים:*\n\n{status}", parse_mode="Markdown", reply_markup=markup)

@bot.callback_query_handler(func=lambda call: call.data.startswith('rem_'))
def handle_reminder_callback(call):
    # call.data formats: rem_done_{id}, rem_snooze_{mins}_{id}
    data = call.data.split('_')
    action = data[1]
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if action == 'done':
                reminder_id = data[2]
                cur.execute("UPDATE reminders SET status = 'completed' WHERE id = %s", (reminder_id,))
                bot.answer_callback_query(call.id, "סומן כבוצע!")
                # Update message to show it's done
                new_text = call.message.text + "\n\n✅ *בוצע*"
                bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, 
                                      text=new_text, parse_mode="Markdown", reply_markup=None)
            elif action == 'snooze':
                mins = int(data[2])
                reminder_id = data[3]
                
                # Update the reminder to a future time and set status back to pending
                cur.execute("""
                    UPDATE reminders 
                    SET remind_at = CURRENT_TIMESTAMP + (%s * interval '1 minute'), status = 'pending' 
                    WHERE id = %s
                """, (mins, reminder_id))
                
                bot.answer_callback_query(call.id, f"נדחה ב-{mins} דקות")
                # Update message to show it's snoozed
                snooze_text = f"נדחה ב-{mins} דקות"
                if mins == 1440: snooze_text = "נדחה למחר"
                elif mins == 60: snooze_text = "נדחה בשעה"
                
                new_text = call.message.text + f"\n\n⏳ *{snooze_text}*"
                bot.edit_message_text(chat_id=call.message.chat.id, message_id=call.message.message_id, 
                                      text=new_text, parse_mode="Markdown", reply_markup=None)
        conn.commit()
    except Exception as e:
        logger.error(f"Callback error: {e}")
        bot.answer_callback_query(call.id, "אירעה שגיאה")
    finally:
        conn.close()

@bot.message_handler(content_types=['text', 'photo', 'voice'])
def handle_message(message):
    user_id = str(message.chat.id)
    user_text = message.text or message.caption or ""
    images = []

    bot.send_chat_action(message.chat.id, 'typing')

    try:
        # Handle voice
        if message.voice:
            file_info = bot.get_file(message.voice.file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            voice_path = f"temp_voice_{user_id}.ogg"
            with open(voice_path, "wb") as new_file:
                new_file.write(downloaded_file)
            
            try:
                transcript = transcribe_ogg(voice_path)
                user_text = (user_text + " " + transcript).strip()
                logger.info(f"Transcribed voice: {transcript}")
            except Exception as e:
                logger.error(f"Voice transcription error: {e}")
                bot.reply_to(message, "מצטער, לא הצלחתי להבין את ההודעה הקולית.")
                return
            finally:
                if os.path.exists(voice_path):
                    os.remove(voice_path)

        # Handle photo
        if message.photo:
            file_info = bot.get_file(message.photo[-1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            photo_path = f"temp_photo_{user_id}.jpg"
            with open(photo_path, "wb") as new_file:
                new_file.write(downloaded_file)
            images.append(Image(filepath=photo_path))

        if not user_text and not images:
            user_text = "Here is a message."

        logger.info(f"Received message from {user_id}: {user_text}")

        agent = create_agent(user_id)
        if images:
            response = agent.run(user_text, images=images)
        else:
            response = agent.run(user_text)
            
        bot.reply_to(message, response.content)

    except Exception as e:
        logger.error(f"Agent error: {e}")
        bot.reply_to(message, "Sorry, I encountered an error processing your request.")
    finally:
        # Cleanup photos
        if message.photo:
            photo_path = f"temp_photo_{user_id}.jpg"
            if os.path.exists(photo_path):
                os.remove(photo_path)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    logger.info("Initializing database...")
    init_db()
    
    logger.info("Starting Flask server for OAuth callbacks...")
    threading.Thread(target=start_flask, daemon=True).start()
    
    logger.info("Starting background scheduler...")
    start_scheduler()
    
    logger.info("Starting Telegram Bot Polling...")
    bot.infinity_polling()
