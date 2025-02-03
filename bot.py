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
    "max_games_per_user": int(os.getenv("MAX_GAMES_PER_USER", 3)),  # Максимальное количество сохраненных игр
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
        max_retries = 3
        retry_delay = 1  # seconds

        for attempt in range(max_retries):
            try:
                # Try to remove the database file if it exists and is locked
                if attempt > 0 and os.path.exists(self.db_path):
                    try:
                        os.remove(self.db_path)
                        logger.info("Removed locked database file")
                    except Exception as e:
                        logger.warning(f"Could not remove database file: {e}")

                async with aiosqlite.connect(self.db_path, timeout=20) as conn:
                    # Enable foreign keys and configure SQLite
                    await conn.execute("PRAGMA foreign_keys = ON")
                    await conn.execute("PRAGMA journal_mode = WAL")
                    await conn.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout

                    # Drop existing tables if they exist
                    await conn.execute("DROP TABLE IF EXISTS users")
                    await conn.execute("DROP TABLE IF EXISTS game_sessions")

                    # First create game_sessions table
                    await conn.execute('''
                        CREATE TABLE game_sessions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER NOT NULL,
                            context TEXT DEFAULT '',
                            history TEXT DEFAULT '',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')

                    # Then create users table with foreign key reference
                    await conn.execute('''
                        CREATE TABLE users (
                            user_id INTEGER PRIMARY KEY,
                            current_game_id INTEGER,
                            saw_welcome BOOLEAN DEFAULT 0,
                            FOREIGN KEY (current_game_id) 
                            REFERENCES game_sessions (id)
                            ON DELETE SET NULL
                        )
                    ''')

                    await conn.commit()
                    logger.info("Database tables recreated successfully")
                    return  # Success, exit the function

            except Exception as e:
                logger.error(f"Database initialization error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise  # Re-raise the last exception if all retries failed
    async def cleanup_old_games(self, user_id: int):
        """Delete old game sessions, keeping only the most recent ones"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Get all game sessions for user, ordered by creation time
                cursor = await conn.execute('''
                    SELECT id FROM game_sessions 
                    WHERE user_id = ? 
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                ''', (user_id, CONFIG["max_games_per_user"]))

                old_games = await cursor.fetchall()
                if old_games:
                    # Delete old games
                    old_game_ids = [game[0] for game in old_games]
                    await conn.execute(
                        'DELETE FROM game_sessions WHERE id IN ({})'.format(
                            ','.join('?' * len(old_game_ids))
                        ),
                        old_game_ids
                    )
                    await conn.commit()
                    logger.info(f"Cleaned up {len(old_game_ids)} old games for user {user_id}")
        except Exception as e:
            logger.error(f"Error cleaning up old games: {e}")
            raise
    async def create_new_game(self, user_id: int) -> int:
        """Create a new game session and return its ID"""
        try:
            # First, cleanup old games
            await self.cleanup_old_games(user_id)

            async with aiosqlite.connect(self.db_path) as conn:
                # Insert new game session
                cursor = await conn.execute(
                    'INSERT INTO game_sessions (user_id, context, history) VALUES (?, ?, ?)',
                    (user_id, '', '')
                )
                game_id = cursor.lastrowid

                # Update user's current game
                await conn.execute(
                    'INSERT OR REPLACE INTO users (user_id, current_game_id) VALUES (?, ?)',
                    (user_id, game_id)
                )
                await conn.commit()
                return game_id
        except Exception as e:
            logger.error(f"Error creating new game: {e}")
            return None
    async def get_user_context(self, user_id: int) -> Dict:
        """Get current game context for user"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                # Get current game session
                cursor = await conn.execute('''
                    SELECT g.context, g.history, u.saw_welcome 
                    FROM users u 
                    LEFT JOIN game_sessions g ON u.current_game_id = g.id 
                    WHERE u.user_id = ?
                ''', (user_id,))
                result = await cursor.fetchone()

                return {
                    "context": result[0] if result and result[0] else "",
                    "history": result[1].split('|') if result and result[1] else [],
                    "saw_welcome": bool(result[2]) if result else False
                }
        except Exception as e:
            logger.error(f"Database error in get_user_context: {e}")
            return {"context": "", "history": [], "saw_welcome": False}

    async def update_user_context(self, user_id: int, context: str, history: List[str], saw_welcome: bool = None):
        """Update current game context"""
        try:
            async with aiosqlite.connect(self.db_path) as conn:
                history_str = '|'.join(history[-CONFIG["max_history_length"]:])

                # Update game session
                await conn.execute('''
                    UPDATE game_sessions 
                    SET context = ?, history = ? 
                    WHERE id = (SELECT current_game_id FROM users WHERE user_id = ?)
                ''', (context, history_str, user_id))

                if saw_welcome is not None:
                    await conn.execute(
                        'UPDATE users SET saw_welcome = ? WHERE user_id = ?',
                        (saw_welcome, user_id)
                    )

                await conn.commit()
        except Exception as e:
            logger.error(f"Database error in update_user_context: {e}")
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

# Классы для работы с контентом
class StoryParser:
    @staticmethod
    def parse_response(text: str) -> tuple:
        try:
            # Normalize line endings and clean up text
            text = text.replace('\r\n', '\n').strip()

            # First try to find sections using markers
            description = ""
            options = []

            # Find description section
            desc_start = text.find("[ОПИСАНИЕ]")
            if desc_start != -1:
                desc_end = text.find("[ВАРИАНТЫ]", desc_start)
                if desc_end == -1:
                    desc_end = len(text)
                description = text[desc_start + len("[ОПИСАНИЕ]"):desc_end].strip()

            # Find options section
            opt_start = text.find("[ВАРИАНТЫ]")
            if opt_start != -1:
                options_text = text[opt_start + len("[ВАРИАНТЫ]"):].strip()

                # Split options by newlines and numbers
                lines = options_text.split('\n')
                for line in lines:
                    # Clean up the line
                    line = re.sub(r'[*_~`]', '', line.strip())
                    line = re.sub(r'^\d+\.?\s*', '', line)
                    line = re.sub(r'^\s*[•\-★]\s*', '', line)

                    # Split by pipe if present
                    if '|' in line:
                        parts = [p.strip() for p in line.split('|')]
                        options.extend(p for p in parts if p and not p.isspace())
                    elif line and not line.isspace():
                        options.append(line)

            # If no sections found, try to split by empty lines
            if not description and not options:
                parts = [p.strip() for p in text.split('\n\n') if p.strip()]
                if len(parts) >= 2:
                    description = parts[0]
                    options = [p for p in parts[1:] if p and not p.isspace()][:3]

            # Clean up and format options
            # Clean up and format options
            cleaned_options = []
            for opt in options[:3]:
                try:
                    opt = opt.strip()
                    # Учитываем, что "⚡ " добавляет 3 символа
                    if len(opt) > 31:
                        opt = opt[:31]  # 31 + 3 (эмодзи) = 34 символа
                    if opt and not opt.isspace():
                        cleaned_options.append(f"⚡ {opt}")
                except Exception as e:
                    logger.error(f"Error processing option: {e}")
            # Add default options if needed
            default_options = ["Продолжить разведку", "Поискать припасы", "Найти укрытие"]
            while len(cleaned_options) < 3:
                cleaned_options.append(f"⚡ {default_options[len(cleaned_options) - 1]}")

            # If no valid description found, use default
            if not description:
                description = "Вы оказались в неизвестной ситуации. Нужно принять решение."

            return description, cleaned_options

        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return "Произошла ошибка при обработке ответа.", ["🔄 Начать заново"]

class StoryGenerator:
    # Class-level model instance
    model = genai.GenerativeModel('gemini-pro')

    # Base prompt template
    STORY_PROMPT_TEMPLATE = """[ИНСТРУКЦИЯ ДЛЯ СЦЕНАРИСТА ИГРЫ]
Вы - опытный сценарист интерактивного квеста в жанре survival-horror (рейтинг 16+).
Ваша задача - создавать последовательные, атмосферные эпизоды истории выживания.

||| СТРУКТУРА ПОВЕСТВОВАНИЯ |||
1. **Преемственность сюжета**:
   • Каждый эпизод должен логически следовать из предыдущих событий
   • Учитывать предыдущие выборы игрока
   • Поддерживать общую линию выживания в городе

2. **Атмосфера и окружение**:
   • Время суток и погода влияют на происходящее
   • Локации связаны между собой (улицы, здания, подвалы)
   • Встречи с другими выжившими или следы их присутствия
   • Звуки и запахи создают объемную картину

3. **Ключевые элементы сцены**:
   • Динамическое окружение (движение, изменения)
   • Визуальные детали (предметы, следы, знаки)
   • Звуковой фон (ambient, тревожные звуки)
   • Эмоциональное состояние персонажа
   • Намеки на возможные пути развития

4. **Структура описания** (4-6 предложений):
   • Вступление: общая картина локации
   • Детали: особенности окружения
   • Действие: что происходит сейчас
   • Ощущения: реакция персонажа
   • Подсказка: важный элемент для выбора

5. **Варианты действий**:
   • СТРОГО 3 варианта, до 30 символов каждый
   • Каждый вариант должен:
     - Быть логичным продолжением ситуации
     - Вести к разным последствиям
     - Использовать формат: [Глагол] + [Объект/Направление]

||| ВАЖНЫЕ ЭЛЕМЕНТЫ |||
• Постоянные угрозы:
  - Зараженные (хрипы, движения, следы)
  - Опасное окружение (темнота, завалы, пожары)
  - Нехватка ресурсов (еда, вода, медикаменты)

• Ключевые темы:
  - Выживание и поиск безопасности
  - Поиск информации о происходящем
  - Помощь другим выжившим
  - Сбор необходимых ресурсов

||| ЗАПРЕТЫ |||
× Нарушение преемственности сюжета
× Явные описания ранений и смертей
× Анатомические подробности
× Сцены насилия
× Неоправданная жестокость

||| ФОРМАТ ОТВЕТА |||
[ОПИСАНИЕ]
<Описание текущей ситуации>

[ВАРИАНТЫ]
1. <Вариант1>
2. <Вариант2>
3. <Вариант3>"""

    # Generation config
    generation_config = {
        "temperature": 0.85,  # Increased for more creative but still controlled responses
        "top_p": 0.9,        # Slightly increased for more narrative variety
        "top_k": 40,
        "max_output_tokens": 1024,
        "candidate_count": 1  # Generate single, focused response
    }

    # Content patterns
    forbidden_patterns = [
        r'\b(убийств[оа]|труп[ыа]|расчленени[ея])\b',
        r'\b(пытк[иа]|издевательств[оа])\b',
        r'\b(суицид|самоубийств[оа])\b',
        r'\b(кровь|кости|плоть)\b',
        r'\b(топор|нож|пистолет)\b',
        r'\b(умер|погиб|сдох)\b',
        r'\b(рана|перелом|травма)\b'
    ]

    allowed_patterns = [
        r'\b(спрятался|укрылся|затаился)\b',
        r'\b(царапина|ушиб|ссадина)\b',
        r'\b(опасность|угроза|риск)\b',
        r'\b(шорох|скрип|хрип)\b',
        r'\b(тень|силуэт|фигура)\b',
        r'\b(шаги|дыхание|стук)\b'
    ]

    # Safety settings
    safety_settings = [
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        }
    ]

    @staticmethod
    async def generate_story(db: Database, user_id: int, choice: str = "") -> dict:
        """Generate a story response based on user context and choice"""
        try:
            user_data = await db.get_user_context(user_id)

            # Format history with arrow separators for better readability
            history_text = ' ➜ '.join(user_data['history']) if user_data['history'] else ""

            # Default starting context if none exists
            current_context = user_data['context'] if user_data['context'] else '''
Начало истории: Сумерки. Ты просыпаешься от странных звуков за окном. 
Телефон и интернет не работают, а на улицах слышны сирены и крики. 
В последних новостях говорили о какой-то эпидемии и призывали сохранять спокойствие.
Нужно разобраться в происходящем и найти безопасное место.'''

            # Format player's choice with timestamp
            choice_text = f"ВЫБОР ИГРОКА ({len(user_data['history']) + 1}): {choice}" if choice else ""

            # Combine template with dynamic content and additional context
            prompt = f"""{StoryGenerator.STORY_PROMPT_TEMPLATE}

||| ТЕКУЩИЙ КОНТЕКСТ |||
{current_context}

||| ИСТОРИЯ ДЕЙСТВИЙ |||
{history_text}

||| ВЫБОР ИГРОКА |||
{choice_text}

||| НАПОМИНАНИЕ |||
• Сохраняйте преемственность сюжета
• Учитывайте предыдущие действия игрока
• Создавайте атмосферные, но не пугающие описания
• Давайте логичные варианты действий"""

            try:
                # Generate response using class-level model with safety settings
                response = await asyncio.to_thread(
                    lambda: StoryGenerator.model.generate_content(
                        prompt,
                        generation_config=StoryGenerator.generation_config,
                        safety_settings=StoryGenerator.safety_settings
                    )
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

                # Parse the response
                story_text, options = StoryParser.parse_response(response.text)

                # Update user context with new story and history
                new_history = user_data['history'] + [choice] if choice else user_data['history']
                await db.update_user_context(user_id, story_text, new_history)

                return {
                    "text": story_text,
                    "options": options
                }

            except Exception as e:
                logger.error(f"Gemini API error: {str(e)}")
                raise

        except Exception as e:
            logger.error(f"Story generation failed: {str(e)}")
            return {
                "text": "⚠️ Сервис временно недоступен. Попробуйте через несколько минут.",
                "options": ["🔄 Начать заново"]
            }

def get_base_keyboard():
    """Create persistent keyboard with main game controls"""
    keyboard = [
        [KeyboardButton("🎮 Новая игра"), KeyboardButton("❓ Помощь")],
        [KeyboardButton("📊 Статус"), KeyboardButton("🔄 Рестарт")]
    ]
    # Делаем клавиатуру постоянной и неизменяемой
    return ReplyKeyboardMarkup(
        keyboard,
        resize_keyboard=True,
        is_persistent=True,
        one_time_keyboard=False,
        input_field_placeholder="Выберите действие..."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help information"""
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
    await update.message.reply_text(
        help_text,
        parse_mode="Markdown",
        reply_markup=get_base_keyboard()
    )
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start a new game session"""
    db = context.bot_data['db']
    user_id = update.effective_user.id
    user_data = await db.get_user_context(user_id)

    # Create new game session
    await db.create_new_game(user_id)

    if not user_data["saw_welcome"]:
        welcome_text = """🧟 *Добро пожаловать в Зомби-Апокалипсис!*

Город охвачен эпидемией. Люди превращаются в зомби, а вы должны выжить.
Используйте смекалку, избегайте опасности и помогайте другим выжившим.

Команды:
🎮 Новая игра - Начать новую историю
❓ Помощь - Показать помощь
📊 Статус - Ваш прогресс
🔄 Рестарт - Начать заново

Удачи! И помните: главное - выжить! 🎯"""
        # Send welcome message with persistent keyboard
        await update.message.reply_text(
            welcome_text,
            parse_mode="Markdown",
            reply_markup=get_base_keyboard()
        )
        # Update user's welcome status
        await db.update_user_context(user_id, "", [], True)
    else:
        # Send a brief message for subsequent starts
        await update.message.reply_text(
            "🎮 Начинаю новую игру...",
            reply_markup=get_base_keyboard()
        )

    # Generate and send the first story response
    response = await StoryGenerator.generate_story(db, user_id)
    await send_response(update, response)
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current game status"""
    db = context.bot_data['db']
    user_id = update.effective_user.id
    user_data = await db.get_user_context(user_id)

    if not user_data['history']:
        await update.message.reply_text(
            "🎯 Вы еще не начали игру. Используйте /start чтобы начать!",
            reply_markup=get_base_keyboard()
        )
        return

    history = user_data['history']
    actions_text = "Нет действий"
    if history:
        actions_text = "➜ " + "\n➜ ".join(history[-3:])

    status_text = f"""📊 *Игровая статистика*

*🌆 Текущая ситуация:*
`{user_data['context'][:200]}...`

▰▰▰▰▰▰▰▰▰▰

*🎯 Последние действия:*
`{actions_text}`

▰▰▰▰▰▰▰▰▰▰

*📈 Прогресс:* `{len(history)} действий`"""
    await update.message.reply_text(
        status_text,
        parse_mode="Markdown",
        reply_markup=get_base_keyboard()
    )
async def restart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Restart the game"""
    db = context.bot_data['db']
    user_id = update.effective_user.id

    # Create new game session
    await db.create_new_game(user_id)

    # Send restart message with persistent keyboard
    await update.message.reply_text(
        "🔄 Игра перезапущена! Начинаем заново...",
        reply_markup=get_base_keyboard()
    )

    # Generate and send the first story response
    response = await StoryGenerator.generate_story(db, user_id)
    await send_response(update, response)
async def send_response(update: Update, response: dict, message=None):
    """Send or edit message with story response"""
    try:
        # Format the story text with rich styling
        content = f"""*🌆 Ситуация:*
`{response['text']}`

▰▰▰▰▰▰▰▰▰▰

*💭 Выберите действие:*"""
        # Create buttons with improved layout
        keyboard = []
        for idx, option in enumerate(response['options']):
            callback_data = f"choice_{idx + 1}"
            keyboard.append([InlineKeyboardButton(option, callback_data=callback_data)])

        # Create inline keyboard and always include base keyboard
        inline_markup = InlineKeyboardMarkup(keyboard) if keyboard else None

        if message and hasattr(message, 'edit_text'):
            try:
                return await message.edit_text(
                    text=content,
                    reply_markup=inline_markup,
                    parse_mode="Markdown"
                )
            except Exception as edit_error:
                logger.error(f"Failed to edit message: {edit_error}")
                if hasattr(update, 'message'):
                    return await update.message.reply_text(
                        text=content,
                        reply_markup=inline_markup,
                        parse_mode="Markdown"
                    )
        elif hasattr(update, 'message'):
            return await update.message.reply_text(
                text=content,
                reply_markup=inline_markup,
                parse_mode="Markdown"
            )
        else:
            logger.error("No valid message object found")
            return None

    except Exception as e:
        logger.error(f"Error in send_response: {e}")
        try:
            if hasattr(update, 'message'):
                return await update.message.reply_text(
                    "⚠️ Произошла ошибка при отправке сообщения. Попробуйте еще раз.",
                    reply_markup=get_base_keyboard()
                )
        except Exception as fallback_error:
            logger.error(f"Failed to send error message: {fallback_error}")
            return None
async def handle_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline keyboard button choices"""
    try:
        db = context.bot_data['db']
        query = update.callback_query
        user_id = query.from_user.id

        # Get the original option text from the button that was clicked
        choice = query.message.reply_markup.inline_keyboard[int(query.data.split('_')[1]) - 1][0].text
        # Remove emoji and extra spaces from choice
        choice = re.sub(r'[⚡🔸🔄]\s*', '', choice).strip()

        await query.answer()
        response = await StoryGenerator.generate_story(db, user_id, choice)
        await send_response(update, response, message=query.message)
    except Exception as e:
        logger.error(f"Error handling choice: {e}")
        # В случае ошибки показываем клавиатуру
        if hasattr(update, 'callback_query'):
            await update.callback_query.message.reply_text(
                "⚠️ Произошла ошибка. Попробуйте еще раз:",
                reply_markup=get_base_keyboard()
            )
async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle keyboard button presses"""
    try:
        text = update.message.text
        # Всегда отправляем клавиатуру с ответом
        reply_markup = get_base_keyboard()

        if text == "🎮 Новая игра":
            await start(update, context)
        elif text == "❓ Помощь":
            await help_command(update, context)
        elif text == "📊 Статус":
            await status_command(update, context)
        elif text == "🔄 Рестарт":
            await restart_command(update, context)
        else:
            # Если получили неизвестную команду, показываем подсказку
            await update.message.reply_text(
                "Выберите действие из меню ниже:",
                reply_markup=reply_markup
            )
    except Exception as e:
        logger.error(f"Error handling button press: {e}")
        # В случае ошибки всегда показываем клавиатуру
        if hasattr(update, 'message'):
            await update.message.reply_text(
                "Произошла ошибка. Попробуйте еще раз:",
                reply_markup=get_base_keyboard()
            )
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors in the bot"""
    error = context.error
    logger.error(f"Ошибка: {error}")

    # Handle message deletion errors silently
    if "Message to delete not found" in str(error):
        return

    try:
        # Check if update exists and has a message
        if update and update.message:
            await update.message.reply_text(
                "⚠️ Произошла непредвиденная ошибка",
                reply_markup=get_base_keyboard()
            )
        elif update and update.callback_query:
            await update.callback_query.answer(
                "⚠️ Произошла ошибка. Попробуйте еще раз."
            )
            await update.callback_query.message.reply_text(
                "Выберите действие:",
                reply_markup=get_base_keyboard()
            )
    except Exception as e:
        logger.error(f"Error in error handler: {e}")

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