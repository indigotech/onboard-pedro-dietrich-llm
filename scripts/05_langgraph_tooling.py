from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, BaseMessageChunk, ToolMessageChunk
from langgraph.graph.graph import CompiledGraph

from chat_config import *
from chat_history import ChatHistory
from db import init_db


def query_llm(agent: CompiledGraph, chat_history: ChatHistory) -> None:
    result = agent.invoke({'messages': chat_history.messages})

    message_count = len(chat_history.messages)
    new_messages = (result.get('messages', []))[message_count:]

    print(text_colors['blue2'], end='', flush=True)
    for message in new_messages:
        if isinstance(message, AIMessage):
            if message.tool_calls:
                print(text_colors['violet2'], end='')
                for tool_call in message.tool_calls:
                    print(f'Search tool call: {tool_call["args"]["search_input"]}\n', flush=True)
            if message.content:
                print(f'{text_colors["blue2"]}{message.content}\n', flush=True)

        chat_history.add_message(message)


def query_llm_stream(agent: CompiledGraph, chat_history: ChatHistory) -> None:
    message_buffer = ''
    gathered = BaseMessageChunk(content='', type='')
    current_step = 0

    for message, metadata in agent.stream({'messages': chat_history.messages}, stream_mode='messages'):
        step = metadata['langgraph_step'] # type: ignore

        if current_step != step:
            if isinstance(gathered, AIMessageChunk):
                chat_history.add_message(AIMessage(content=gathered.content, tool_calls=gathered.tool_calls))
            elif isinstance(gathered, ToolMessageChunk):
                chat_history.add_message(ToolMessage(content=gathered.content))

            current_step = step

            if isinstance(message, BaseMessageChunk):
                gathered = message
            if isinstance(message, AIMessageChunk):
                for tool_call in message.tool_calls:
                    print(f'{text_colors["violet2"]}Using {tool_call["name"]} tool...\n', flush=True)
            elif isinstance(message, ToolMessage):
                chat_history.add_message(message)
                gathered = BaseMessageChunk(content='', type='')
        else:
            if isinstance(message, BaseMessageChunk):
                gathered = gathered + message

        if isinstance(message, AIMessageChunk) and message.content:
            print(f'{text_colors["blue2"]}{message.content}', end='', flush=True)
            message_buffer += str(message.content)

    if isinstance(gathered, AIMessageChunk):
        chat_history.add_message(AIMessage(content=gathered.content, tool_calls=gathered.tool_calls))
    elif isinstance(gathered, ToolMessageChunk):
        chat_history.add_message(ToolMessage(content=gathered.content))

    print('\n')


def chat(agent: CompiledGraph, chat_history: ChatHistory, stream: bool) -> None:
    user_input = input(f'{text_colors["green2"]}User ("quit" to exit): ')
    print()
    while user_input != 'quit' and user_input != 'exit':
        chat_history.add_message(HumanMessage(content=user_input))

        query_llm_stream(agent, chat_history) if stream else query_llm(agent, chat_history)

        user_input = input(f'{text_colors["green2"]}User: ')
        print()


def main():
    load_dotenv()
    args = get_arguments()
    conn = init_db()

    if args.stream:
        print('Using streaming mode.\n')
    else:
        print('Streaming mode disabled.\n')

    agent = create_agent(args.vendor, [web_search])
    chat_history = ChatHistory(args.chatid, conn)

    chat(agent, chat_history, args.stream)
    chat_history.save_messages()

    conn.close()


main()
