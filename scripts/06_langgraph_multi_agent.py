from dotenv import load_dotenv
from langchain_core.messages import AIMessageChunk, ToolMessageChunk, ToolCall
from langchain_core.runnables import Runnable

from chat_config import *
from chat_history import ChatHistory
from db import init_db


def create_agent_call_tool(agent: CompiledGraph, agent_name: str, description: str | None = None) -> BaseTool:
    name = f'transfer_to_{agent_name}'
    description = description or f'Delegate task to the {agent_name}.'

    @tool(name, description=description)
    def call_tool(query: str) -> str:
        response = agent.invoke({'messages': [HumanMessage(content=query)]})
        message: BaseMessage = response['messages'][-1]
        return str(message.content)

    return call_tool


def create_agents(vendor: str) -> tuple[Runnable, dict[str, BaseTool]]:
    llm_model = get_chat_model(vendor)

    research_agent = create_react_agent(
        name='research_agent',
        model=llm_model,
        tools=[web_search],
        prompt=(
            'You are a research agent. You only perform web research tasks, using the web_search tool.\n'
            'As soon as you finish the search, you return to the supervisor the search results.\n'
            'Do not include extra text in the search results.'
        )
    )
    calculator_agent = create_react_agent(
        name='calculator_agent',
        model=llm_model,
        tools=[add, subtract, multiply],
        prompt=(
            'You are a calculator agent. Do only sum, subtraction and multiplication, and nothing else.\n'
            'Respond directly to the supervisor, and do not include any text other than the task results.'
        )
    )
    writer_agent = create_react_agent(
        name='writer_agent',
        model=llm_model,
        tools=[],
        prompt=(
            'You are a writer agent. You should only generate text for the response.\n'
            'Respond directly to the supervisor.'
        )
    )

    call_research_agent = create_agent_call_tool(
        research_agent,
        agent_name='research_agent',
        description='Assign task to the research agent.'
    )
    call_calculator_agent = create_agent_call_tool(
        calculator_agent,
        agent_name='calculator_agent',
        description='Assing task to the calculator agent. Assign all math operations to the calculator agent, always.'
    )
    call_writer_agent = create_agent_call_tool(
        writer_agent,
        agent_name='writer_agent',
        description='Assing task to the writer agent. Always use the writer agent to write the responses.'
    )

    supervisor_runnable = llm_model.bind_tools([call_research_agent, call_calculator_agent, call_writer_agent])
    return supervisor_runnable, {
        'transfer_to_research_agent': call_research_agent,
        'transfer_to_calculator_agent': call_calculator_agent,
        'transfer_to_writer_agent': call_writer_agent
    }


def query_llm(chat_history: ChatHistory, chain: Runnable) -> list[ToolCall]:
    output_message = chain.invoke(chat_history.messages)
    tool_calls: list[ToolCall] = []

    if isinstance(output_message, BaseMessage):
        chat_history.add_message(output_message)
    if isinstance(output_message, AIMessage):
        ai_response = str(output_message.content)
        if ai_response:
            print(f'{text_colors["blue"]}{ai_response}\n')
        tool_calls = output_message.tool_calls

    return tool_calls

def query_llm_stream(chat_history: ChatHistory, chain: Runnable) -> list[ToolCall]:
    ai_response = ''
    gathered = AIMessageChunk(content='')
    is_first = True

    for chunk in chain.stream(chat_history.messages):
        ai_response += str(chunk.content)
        if is_first:
            print(text_colors['blue'], end='', flush=True)
            gathered = chunk
            is_first = False
        else:
            gathered = gathered + chunk
        print(chunk.content, end='', flush=True)

    if gathered.content: print('\n')

    tool_calls = gathered.tool_calls

    if isinstance(gathered, AIMessageChunk):
        chat_history.add_message(AIMessage(content=gathered.content, tool_calls=gathered.tool_calls))
    elif isinstance(gathered, ToolMessageChunk):
        chat_history.add_message(ToolMessage(content=gathered.content))

    return tool_calls


def chat(chat_history: ChatHistory, chain: Runnable, subagent_calls: dict[str, BaseTool], stream: bool) -> None:
    user_input = input(f'\n{text_colors["green2"]}User ("quit" to exit): ')
    print(flush=True)
    while user_input != 'quit' and user_input != 'exit':
        chat_history.add_message(HumanMessage(content=user_input))
        agent_loop_count = 0
        while agent_loop_count < 5:
            tool_calls = query_llm_stream(chat_history, chain) if stream else query_llm(chat_history, chain)
            if len(tool_calls) == 0: break

            for tool_call in tool_calls:
                print(f'{text_colors["violet2"]}Tool call: {tool_call["name"]}\n')
                selected_tool = subagent_calls[tool_call['name']]
                tool_response: ToolMessage = selected_tool.invoke(tool_call)
                chat_history.add_message(tool_response)
            agent_loop_count += 1

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

    chat_history = ChatHistory(args.chatid, conn)

    supervisor, subagent_calls = create_agents(args.vendor)

    chat(chat_history, supervisor, subagent_calls, args.stream)
    chat_history.save_messages()

    conn.close()


main()
