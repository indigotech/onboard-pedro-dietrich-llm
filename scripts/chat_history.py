import datetime as dt
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, ToolMessage

from chat_config import *
from db import *

class ChatHistory(BaseChatMessageHistory):
    def __init__(self, chat_id: str, conn: sqlite3.Connection):
        self.chat_id: str = chat_id
        self.conn: sqlite3.Connection = conn

        db_messages, db_context = fetch_history(self.conn, self.chat_id)
        self.messages: list[BaseMessage] = db_messages
        self.new_messages: list[MessageData] = []
        self.context: str = db_context

        self.initialize_chat()

    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)

        if isinstance(message, ToolMessage) or not message.content: return
        self.new_messages.append(MessageData(dt.datetime.now(), get_message_role(message), str(message.content)))

    def update_context(self, new_context: str) -> None:
        self.context = new_context

    def save_messages(self) -> None:
        for m in self.new_messages:
            save_message(self.conn, self.chat_id, m[0], m[1], m[2])

    def save_context(self) -> None:
        save_context(self.conn, self.chat_id, str(self.context))

    def clear(self) -> None:
        self.messages = []
        self.new_messages = []

    def initialize_chat(self) -> None:
        if len(self.messages) == 0:
            self.chat_id = create_new_chat(self.conn)
            print(f'{text_colors["yellow2"]}Chat ID: {self.chat_id}\n')
            self.add_message(SystemMessage(content='You are a helpful assistant.'))
        else:
            print(f'{text_colors["yellow2"]}Chat ID: {self.chat_id}\n')
            for m in self.messages:
                if isinstance(m, AIMessage) and m.content:
                    print(f'{text_colors["blue2"]}{m.content}\n')
                elif isinstance(m, HumanMessage):
                    print(f'{text_colors["green2"]}{m.content}\n')
