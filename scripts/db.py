import datetime
import sqlite3
import uuid
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect('chat_history.db')
    conn.cursor().execute(
        """
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                context TEXT
            );
        """
    )
    conn.cursor().execute(
        """
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                chat_id TEXT,
                time TIMESTAMP,
                role TEXT,
                content TEXT,
                FOREIGN KEY(chat_id) REFERENCES chats(chat_id)
            );
        """
    )
    conn.commit()
    return conn

def create_new_chat(conn: sqlite3.Connection) -> str:
    chat_id = str(uuid.uuid4())
    cursor = conn.cursor()
    cursor.execute(
        """
            INSERT INTO chats
            (chat_id, context) VALUES (?, ?);
        """,
        [chat_id, '']
    )
    conn.commit()
    return chat_id

def fetch_history(conn: sqlite3.Connection, chat_id: str) -> tuple[list[BaseMessage], str]:
    cursor = conn.cursor()
    cursor.execute(
        """
            SELECT role, content FROM messages
            WHERE chat_id = ?
            ORDER BY time ASC;
        """,
        [chat_id]
    )
    rows: list[tuple[str, str]] = cursor.fetchall()

    cursor.execute(
        """
            SELECT context FROM chats
            WHERE chat_id = ?;
        """,
        [chat_id]
    )
    context = cursor.fetchone()
    context: str = context[0] if isinstance(context, list) else ''

    chat: list[BaseMessage] = []
    for role, content in rows:
        if role == 'system':
            chat.append(SystemMessage(content))
        elif role == 'user':
            chat.append(HumanMessage(content))
        elif role == 'assistant':
            chat.append(AIMessage(content))
        else:
            raise ValueError(f'Unknown role in DB: {role}')
    return chat, context

def save_message(conn: sqlite3.Connection, chat_id: str, time: datetime.datetime, role: str, content: str) -> None:
    cursor = conn.cursor()
    message_id = str(uuid.uuid4())

    cursor.execute(
        """
            INSERT INTO messages (message_id, chat_id, time, role, content)
            VALUES (?, ?, ?, ?, ?);
        """,
        (message_id, chat_id, time, role, content)
    )
    conn.commit()

def save_context(conn: sqlite3.Connection, chat_id: str, context: str) -> None:
    cursor = conn.cursor()

    cursor.execute(
        """
            UPDATE chats
            SET context = ?
            WHERE chat_id = ?
        """,
        (context, chat_id)
    )

    conn.commit()
