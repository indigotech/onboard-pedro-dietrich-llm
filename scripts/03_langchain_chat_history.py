from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from chat_config import *
from db import *

def chat(chat_history: list[BaseMessage], model: BaseChatModel, stream: bool) -> list[MessageData]:
    new_messages: list[MessageData] = []

    user_input = input('  User ("quit" to exit): ')
    while user_input != 'quit' and user_input != 'exit':
        print()
        chat_history.append(HumanMessage(content=user_input))
        new_messages.append(MessageData(datetime.datetime.now(), 'user', user_input))

        ai_response: str = ''
        if stream:
            for chunk in model.stream(chat_history):
                print(chunk.content, end='', flush=True)
                ai_response += str(chunk.content)
            print('\n')
        else:
            output_message = model.invoke(chat_history)
            print(f'  AI: {output_message.content}', '\n')
            ai_response = str(output_message.content)

        chat_history.append(AIMessage(content=ai_response))
        new_messages.append(MessageData(datetime.datetime.now(), 'assistant', ai_response))

        user_input = input('  User ("quit" to exit): ')
    return new_messages


def main():
    load_dotenv()
    args = get_arguments()
    conn = init_db()

    model = get_chat_model(args.vendor)
    if args.stream:
        print('Using streaming mode.\n')
    else:
        print('Streaming mode disabled.\n')

    chat_id: str = args.chatid
    chat_history: list[BaseMessage] = []
    new_messages: list[MessageData] = []

    if args.chatid:
        chat_history = fetch_history(conn, chat_id)

    if len(chat_history) == 0:
        chat_id = create_new_chat(conn)
        print(f'Chat ID: {chat_id}')
        system_prompt = 'You are a helpful assistant.'
        chat_history.append(SystemMessage(content=system_prompt))
        new_messages.append(MessageData(datetime.datetime.now(), 'system', system_prompt))
    else:
        print(f'Chat ID: {chat_id}')
        print(f'Chat history:')
        for i in range(len(chat_history)):
            if i == 0: continue
            speaker = 'User' if i % 2 == 1 else 'AI'
            print(f'  {speaker}: {chat_history[i].content}\n')

    new_messages.extend(chat(chat_history, model, args.stream))
    for message in new_messages:
        save_message(conn, chat_id, message[0], message[1], message[2])

    conn.close()


main()
