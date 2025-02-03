import os
import re
import logging
from typing import Dict, List
from dotenv import load_dotenv
import aiosqlite
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)
import google.generativeai as genai
import asyncio

# Загрузка переменных окружения
load_dotenv()

# Настройка логгера
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# Уменьшаем уровень логирования для служебных сообщений
for logger_name in ['apscheduler', 'httpx', 'telegram.ext', 'telegram.bot']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Конфигурация
CONFIG = {
    "telegram_token": os.getenv("TELEGRAM_TOKEN"),
    "gemini_api_key": os.getenv("GEMINI_API_KEY"),
    "max_history_length": int(os.getenv("MAX_HISTORY_LENGTH", 5)),
    "db_path": "story_bot.db"
}

# Initialize Gemini
if not CONFIG["gemini_api_key"]:
    raise ValueError("GEMINI_API_KEY environment variable is not set")
genai.configure(api_key=CONFIG["gemini_api_key"])


class Database:
    def __init__(self):
        self.db_path = CONFIG["db_path"]

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as conn:
            # Create the table if it doesn't exist
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    context TEXT,
                    history TEXT,
                    saw_welcome BOOLEAN DEFAULT 0
                )
            ''')

            # Check if saw_welcome column exists
            cursor = await conn.execute("PRAGMA table_info(users)")
            columns = await cursor.fetchall()
            column_names = [column[1] for column in columns]

            # Add saw_welcome column if it doesn't exist
            if 'saw_welcome' not in column_names:
                await conn.execute('ALTER TABLE users ADD COLUMN saw_welcome BOOLEAN DEFAULT 0')

            await conn.commit()
    async def get_user_context(self, user_id: int) -> Dict:
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                cursor = await conn.execute(
                    'SELECT context, history, saw_welcome FROM users WHERE user_id = ?',
                    (user_id,)
                )
                result = await cursor.fetchone()
                return {
                    "context": result[0] if result else "",
                    "history": result[1].split('|') if result and result[1] else [],
                    "saw_welcome": bool(result[2]) if result else False
                } if result else {"context": "", "history": [], "saw_welcome": False}
        except Exception as e:
            logger.error(f"Database error in get_user_context: {e}")
            return {"context": "", "history": [], "saw_welcome": False}
    async def update_user_context(self, user_id: int, context: str, history: List[str], saw_welcome: bool = None):
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                history_str = '|'.join(history[-CONFIG["max_history_length"]:])
                if saw_welcome is not None:
                    await conn.execute(
                        '''INSERT OR REPLACE INTO users (user_id, context, history, saw_welcome)
                           VALUES (?, ?, ?, ?)''',
                        (user_id, context, history_str, saw_welcome)
                    )
                else:
                    await conn.execute(
                        '''INSERT OR REPLACE INTO users (user_id, context, history)
                           VALUES (?, ?, ?)''',
                        (user_id, context, history_str)
                    )
                await conn.commit()
        except Exception as e:
            logger.error(f"Database error in update_user_context: {e}")
            # В случае ошибки пытаемся восстановить предыдущий контекст
            try:
                old_context = await self.get_user_context(user_id)
                logger.info(f"Restored previous context for user {user_id}")
            except Exception as restore_error:
                logger.error(f"Failed to restore context: {restore_error}")
# Инициализация базы данных
async def init_database():
    db = Database()
    await db.init_db()
    return db


# Генерация контента через Gemini
class StoryParser:
    @staticmethod
    def parse_response(text: str) -> tuple:
        try:
            # Split by sections
            sections = text.split("[ВАРИАНТЫ]")
            if len(sections) != 2:
                sections = text.split("[ОПИСАНИЕ]")
                if len(sections) > 1:
                    description = sections[1]
                    options_text = sections[-1]
                else:
                    raise ValueError("Could not find sections")
            else:
                description = sections[0].replace("[ОПИСАНИЕ]", "").strip()
                options_text = sections[1].strip()

            # Clean up description
            description = description.strip()

            # Extract options using more robust pattern matching
            options = []
            lines = options_text.split('\n')
            for line in lines:
                # Remove markdown formatting and special characters
                line = re.sub(r'[*_~`]', '', line)  # Remove markdown formatting
                line = re.sub(r'^\d+\.?\s*', '', line.strip())  # Remove numbering
                line = re.sub(r'^\s*[•\-★]\s*', '', line)  # Remove bullet points

                # Split by pipe if present
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    options.extend(p for p in parts if p and not p.isspace())
                # Add non-empty lines
                elif line and not line.isspace():
                    options.append(line)

            # Clean up and format options
            # Clean up and format options
            cleaned_options = []
            for opt in options[:3]:  # Limit to 3 options
                try:
                    # Clean up the option text
                    opt = re.sub(r'[*_~`]', '', opt.strip())  # Remove markdown
                    opt = re.sub(r'^\d+\.?\s*', '', opt)  # Remove numbers
                    opt = opt.strip()
                    # Add emoji and limit length if option is valid
                    if opt and not opt.isspace() and len(opt) <= 25:
                        cleaned_options.append(f"🔸 {opt}")
                    elif opt and not opt.isspace():
                        cleaned_options.append(f"🔸 {opt[:25]}")
                except Exception as e:
                    logger.error(f"Error processing option: {e}")

            # If we don't have enough valid options, add some defaults
            default_options = ["Продолжить разведку", "Поискать припасы", "Найти укрытие"]
            while len(cleaned_options) < 3:
                cleaned_options.append(f"🔸 {default_options[len(cleaned_options) - 1]}")
            return description, cleaned_options
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return "Произошла ошибка при обработке ответа.", ["🔄 Начать заново"]

class StoryGenerator:
    # Class-level model instance
    model = genai.GenerativeModel('gemini-pro')

    # Forbidden content patterns (строго запрещено)
    forbidden_patterns = [
        r'\b(убийств[оа]|труп[ыа]|расчленени[ея])\b',
        r'\b(пытк[иа]|издевательств[оа])\b',
        r'\b(суицид|самоубийств[оа])\b'
    ]

    # Allowed content patterns (допустимый уровень)
    allowed_patterns = [
        r'\b(отбился|защитился|спрятался)\b',
        r'\b(ранение|царапина|ушиб)\b',
        r'\b(опасность|угроза|риск)\b'
    ]

    # Configure safety settings
    safety_settings = [
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    # Configure generation parameters
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    @staticmethod
    async def generate_story(db: Database, user_id: int, choice: str = "") -> dict:
        user_data = await db.get_user_context(user_id)

        history_text = ' -> '.join(user_data['history']) if user_data['history'] else ""
        current_context = user_data['context'] if user_data['context'] else 'Начало истории: Ты просыпаешься от странных звуков за окном. Телефон и интернет не работают, а на улицах слышны сирены и крики. В новостях говорили о какой-то эпидемии.'
        choice_text = f"ВЫБОР ИГРОКА: {choice}" if choice else ""

        prompt = f"""
        [ВНИМАНИЕ: Это текст для компьютерной игры с рейтингом 12+]
        
        Ты — создатель интерактивных квестов в жанре зомби-выживания. Продолжи историю выживания в городе во время зомби-апокалипсиса.

        **Основной сюжет:**
        Город охвачен эпидемией, превращающей людей в зомби. Игрок должен выжить, избегая прямых столкновений, 
        находя припасы и помогая другим выжившим. Акцент на скрытности и умном использовании ресурсов.
        
        **Допустимый уровень контента:**
        - Зомби присутствуют, но описываются нейтрально ("медленно движущиеся фигуры", "странные люди")
        - Можно убегать, прятаться, отвлекать внимание
        - Легкая самооборона (оттолкнуть, увернуться)
        - Поиск припасов и безопасных мест
        
        **Запрещено:**
        - Кровь и расчленёнка
        - Убийства (даже зомби)
        - Тяжёлые травмы
        
        **Стиль повествования:**
        1. Описание: 2-3 коротких предложения (максимум 150 символов)
        2. Варианты действий: 
           - Точно 3 варианта по 20-25 символов
           - Начинать с глагола
           - Реалистичные решения
           - Акцент на выживании и помощи
        
        **Текущий контекст:**
        {current_context}
        
        **История действий:**
        {history_text}
        
        **Последний выбор:**
        {choice_text}
        
        **Формат ответа:**
        [ОПИСАНИЕ]
        Краткое описание текущей ситуации (2-3 предложения)
        
        [ВАРИАНТЫ]
        1. Действие1 | 2. Действие2 | 3. Действие3
        
        Помни: 
        - Каждое действие должно быть логически связано с предыдущим выбором
        - Избегай жестокости, делай акцент на хитрости и находчивости
        - Поддерживай атмосферу напряжения, но без излишнего страха"""

        try:
            # Generate response using class-level model with safety settings
            response = await asyncio.to_thread(
                StoryGenerator.model.generate_content,
                prompt,
                generation_config=StoryGenerator.generation_config,
                safety_settings=StoryGenerator.safety_settings
            )

            if not response or not response.text:
                raise Exception("Empty response received")

            # Check for forbidden content
            response_text = response.text.lower()
            for pattern in StoryGenerator.forbidden_patterns:
                if re.search(pattern, response_text):
                    logger.warning(f"Found forbidden content matching pattern: {pattern}")
                    return {
                        "text": "⚠️ Контент не соответствует требованиям. Генерирую новый вариант...",
                        "options": ["🔄 Начать заново"]
                    }

            logger.info("Successfully generated response using Gemini")

            # Parse the response
            story_text, options = StoryParser.parse_response(response.text)

            # Update user context
            new_history = user_data['history'] + [choice] if choice else user_data['history']
            await db.update_user_context(user_id, story_text, new_history)

            return {
                "text": story_text,
                "options": options
            }

        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            return {
                "text": "⚠️ Сервис временно недоступен. Попробуйте через несколько минут.",
                "options": []
            }

def get_base_keyboard():
    keyboard = [
        [KeyboardButton("🎮 Новая игра"), KeyboardButton("❓ Помощь")],
        [KeyboardButton("📊 Статус"), KeyboardButton("🔄 Рестарт")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

# Обработчики Telegram
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = context.bot_data['db']
    user_id = update.effective_user.id
    user_data = await db.get_user_context(user_id)

    if not user_data["saw_welcome"]:
        welcome_text = """🧟 *Добро пожаловать в Зомби-Апокалипсис!*

Город охвачен эпидемией. Люди превращаются в зомби, а вы должны выжить.
Используйте смекалку, избегайте опасности и помогайте другим выжившим.

Команды:
🎮 Новая игра - Начать историю
❓ Помощь - Показать помощь
📊 Статус - Ваш прогресс
🔄 Рестарт - Начать заново

Удачи! И помните: главное - выжить! 🎯"""
        # Send welcome message with persistent keyboard
        welcome_msg = await update.message.reply_text(
            welcome_text,
            parse_mode="Markdown",
            reply_markup=get_base_keyboard()
        )
        # Schedule message deletion after 30 seconds
        context.job_queue.run_once(
            lambda _: welcome_msg.delete(),
            30
        )
        # Update user's welcome status
        await db.update_user_context(user_id, "", [], True)
    else:
        # Send a brief message for subsequent starts
        start_msg = await update.message.reply_text(
            "🎮 Начинаю новую игру...",
            reply_markup=get_base_keyboard()
        )
        # Delete message after 3 seconds
        context.job_queue.run_once(
            lambda _: start_msg.delete(),
            3
        )
        await db.update_user_context(user_id, "", [])

    # Generate and send the first story response
    response = await StoryGenerator.generate_story(db, user_id)
    await send_response(update, response)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """🧟 *Руководство по выживанию*

• Внимательно читайте описание ситуации
• Выбирайте действия с умом - от них зависит ваша судьба
• Помните о других выживших - вместе у вас больше шансов
• Избегайте прямых столкновений с зомби
• Ищите припасы и безопасные места

Команды:
/start - Начать новую игру
/help - Показать это сообщение
/status - Показать ваш прогресс
/restart - Начать заново

💡 Совет: Хитрость и осторожность важнее силы!"""
    await update.message.reply_text(help_text, parse_mode="Markdown")
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = context.bot_data['db']
    user_id = update.effective_user.id
    user_data = await db.get_user_context(user_id)

    if not user_data['history']:
        await update.message.reply_text("🎯 Вы еще не начали игру. Используйте /start чтобы начать!")
        return

    history = user_data['history']
    actions_text = "Нет действий"
    if history:
        actions_text = "➜ " + "\n➜ ".join(history[-3:])

    status_text = f"""📊 *Ваш прогресс в игре*

Текущая ситуация:
{user_data['context'][:200]}...

Последние действия:
{actions_text}

Всего сделано выборов: {len(history)}"""
    await update.message.reply_text(status_text, parse_mode="Markdown")

async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = context.bot_data['db']
    user_id = update.effective_user.id
    await db.update_user_context(user_id, "", [])

    # Send temporary message
    restart_msg = await update.message.reply_text("🔄 Игра перезапущена! Начинаем заново...")
    # Delete message after 3 seconds
    context.job_queue.run_once(
        lambda _: restart_msg.delete(),
        3
    )

    response = await StoryGenerator.generate_story(db, user_id)
    await send_response(update, response)
async def handle_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db = context.bot_data['db']
    query = update.callback_query
    user_id = query.from_user.id

    # Get the original option text from the button that was clicked
    choice = query.message.reply_markup.inline_keyboard[int(query.data.split('_')[1]) - 1][0].text

    await query.answer()
    response = await StoryGenerator.generate_story(db, user_id, choice)
    await send_response(update, response, message=query.message)

async def send_response(update: Update, response: dict, message=None):
    try:
        # Format the story text with some styling
        content = f"{response['text']}\n\n💭 Выберите действие:"

        # Create buttons with improved layout
        keyboard = []
        for idx, option in enumerate(response['options']):
            callback_data = f"choice_{idx + 1}"
            keyboard.append([InlineKeyboardButton(option, callback_data=callback_data)])

        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None

        if message:
            try:
                await message.edit_text(
                    text=content,
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
            except Exception as edit_error:
                logger.error(f"Failed to edit message: {edit_error}")
                # Если не удалось отредактировать, пробуем отправить новое
                await update.message.reply_text(
                    text=content,
                    reply_markup=reply_markup,
                    parse_mode="Markdown"
                )
        else:
            await update.message.reply_text(
                text=content,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"Error in send_response: {e}")
        try:
            await update.message.reply_text(
                "⚠️ Произошла ошибка при отправке сообщения. Попробуйте еще раз.",
                reply_markup=get_base_keyboard()
            )
        except Exception as fallback_error:
            logger.error(f"Failed to send error message: {fallback_error}")

async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle keyboard button presses"""
    text = update.message.text
    if text == "🎮 Новая игра":
        await start(update, context)
    elif text == "❓ Помощь":
        await help_command(update, context)
    elif text == "📊 Статус":
        await status_command(update, context)
    elif text == "🔄 Рестарт":
        await restart_command(update, context)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    error = context.error
    logger.error(f"Ошибка: {error}")

    # Handle message deletion errors silently
    if "Message to delete not found" in str(error):
        return

    # Check if update exists and has a message
    if update and update.message:
        await update.message.reply_text("⚠️ Произошла непредвиденная ошибка")
    elif update and update.callback_query:
        await update.callback_query.answer("⚠️ Произошла ошибка. Попробуйте еще раз.")
async def post_init(application):
    # Initialize database
    application.bot_data['db'] = await init_database()

    # Set up bot commands for the command menu
    commands = [
        ("start", "Начать новую игру"),
        ("help", "Показать помощь"),
        ("status", "Показать прогресс"),
        ("restart", "Начать заново")
    ]
    await application.bot.set_my_commands(commands)

def main():
    try:
        app = (
            ApplicationBuilder()
            .token(CONFIG["telegram_token"])
            .post_init(post_init)
            .build()
        )

        # Add handlers
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CommandHandler("status", status_command))
        app.add_handler(CommandHandler("restart", restart_command))
        app.add_handler(CallbackQueryHandler(handle_choice))
        # Add handler for keyboard buttons
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_button))
        app.add_error_handler(error_handler)

        logger.info("Initializing Gemini API...")
        logger.info("Бот запущен с Gemini")

        # Run the bot until Ctrl+C is pressed
        app.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.error(f"Critical error: {e}")
    finally:
        logger.info("Bot stopped")

if __name__ == "__main__":
    main()