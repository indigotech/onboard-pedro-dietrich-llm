from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableWithMessageHistory

from chat_config import get_arguments, get_chat_model
from db import *

def get_role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return 'system'
    elif isinstance(message, HumanMessage):
        return 'user'
    elif isinstance(message, AIMessage):
        return 'assistant'
    else:
        raise ValueError(f'Unsupported message type: {type(message)}')


class ChatHistory(BaseChatMessageHistory):
    def __init__(self, chat_id: str, conn: sqlite3.Connection):
        self.chat_id: str = chat_id
        self.conn: sqlite3.Connection = conn
        self.messages: list[BaseMessage] = fetch_history(self.conn, self.chat_id)
        self.new_messages: list[tuple[int, str, str]] = []
        self.position: int = len(self.messages)

        if self.position == 0:
            self.chat_id = create_new_chat(conn)
            print(f'Chat ID: {self.chat_id}\n')
            self.add_message(SystemMessage(content='You are a helpful assistant.'))
        else:
            print(f'Chat ID: {self.chat_id}\n')
            for i in range(self.position):
                if i == 0: continue
                print(f'  {"User" if i % 2 == 1 else "AI"}: {self.messages[i].content}\n')


    def add_message(self, message: BaseMessage) -> None:
        self.messages.append(message)
        self.new_messages.append((self.position, get_role(message), str(message.content)))
        self.position += 1

    def save_messages(self) -> None:
        for m in self.new_messages:
            save_message(self.conn, self.chat_id, m[0], m[1], m[2])

    def clear(self) -> None:
        self.messages = []
        self.new_messages = []
        self.position = 0


def chat(chat_history: ChatHistory, chain: Runnable, stream: bool) -> None:
    user_input = input('  User ("quit" to exit): ')
    while user_input != 'quit' or user_input == 'exit':
        ai_response: str = ''
        if stream:
            for chunk in chain.stream(user_input, config={'configurable': {'session_id': chat_history.chat_id}}):
                print(chunk.content, end='', flush=True)
                ai_response += str(chunk.content)
            print('\n')
        else:
            output_message = chain.invoke(user_input, config={'configurable': {'session_id': chat_history.chat_id}})
            print(f'  AI: {output_message.content}', '\n')
            ai_response = str(output_message.content)

        user_input = input('  User ("quit" to exit): ')


def main():
    load_dotenv()
    args = get_arguments()
    conn = init_db()

    model = get_chat_model(args.vendor)
    if args.stream:
        print('Using streaming mode.\n')
    else:
        print('Streaming mode disabled.\n')

    chain: Runnable = RunnableWithMessageHistory(
        model,
        lambda chat_id: chat_history if chat_id == chat_history.chat_id else ChatHistory(chat_id, conn)
    )

    chat_history = ChatHistory(args.chatid, conn)
    chat(chat_history, chain, args.stream)
    chat_history.save_messages()

    conn.close()


main()
