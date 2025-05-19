from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, ToolCall
from langchain_core.runnables import Runnable, RunnableWithMessageHistory

from chat_config import *
from chat_history import ChatHistory
from db import init_db


def query_llm(input: str, chat_history: ChatHistory, chain: Runnable) -> list[ToolCall]:
    output_message: AIMessage = chain.invoke(input, config={'configurable': {'session_id': chat_history.chat_id}})
    ai_response = str(output_message.content)
    if ai_response:
        print(f'{text_colors["blue"]}{ai_response}\n')
    tool_calls = output_message.tool_calls

    return tool_calls

def query_llm_stream(input: str, chat_history: ChatHistory, chain: Runnable) -> list[ToolCall]:
    ai_response = ''
    gathered = AIMessageChunk(content='')
    is_first = True

    for chunk in chain.stream(input, config={'configurable': {'session_id': chat_history.chat_id}}):
        ai_response += str(chunk.content)
        if is_first:
            print(text_colors['blue'], end='', flush=True)
            gathered = chunk
            is_first = False
        else:
            gathered = gathered + chunk
        print(chunk.content, end='', flush=True)

    tool_calls = gathered.tool_calls
    if len(tool_calls) > 0 and not ai_response:
        print('Using tool...')
    print('\n')

    return tool_calls


def chat(chat_history: ChatHistory, chain: Runnable, stream: bool) -> None:
    user_input = input(f'{text_colors["green2"]}User ("quit" to exit): ')
    print(flush=True)
    while user_input != 'quit' and user_input != 'exit':
        agent_loop_count = 0
        query_input = user_input
        while agent_loop_count < 5:
            tool_calls = query_llm_stream(query_input, chat_history, chain) if stream else query_llm(query_input, chat_history, chain)
            if len(tool_calls) == 0: break

            for tool_call in tool_calls:
                print(f'{text_colors["violet2"]}Tool call - Search: {tool_call["args"]["search_input"]}\n')
                selected_tool = {'web_search': web_search}[tool_call['name'].lower()]
                tool_response: ToolMessage = selected_tool.invoke(tool_call)
                chat_history.add_message(tool_response)
            query_input = ''
            agent_loop_count += 1

        user_input = input(f'{text_colors["green2"]}User: ')
        print()


def main():
    load_dotenv()
    args = get_arguments()
    conn = init_db()

    model = get_chat_model(args.vendor).bind_tools([web_search])

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
