from dotenv import load_dotenv
from langchain_core.tools import InjectedToolCallId
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing import Annotated

from chat_config import *
from chat_history import ChatHistory
from db import init_db


def create_handoff_tool(*, agent_name: str, description: str | None = None) -> BaseTool:
    name = f'transfer_to_{agent_name}'
    description = description or f'Ask {agent_name} for help.'

    @tool(name, description=description)
    def handoff_tool(state: Annotated[MessagesState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
        tool_message = {
            'role': 'tool',
            'content': f'{text_colors["cyan2"]}Successfully transfered to {agent_name}.{text_colors["blue2"]}',
            'name': name,
            'tool_call_id': tool_call_id
        }
        return Command(goto=agent_name, update={**state, 'messages': state['messages'] + [tool_message]}, graph=Command.PARENT)

    return handoff_tool


def create_system_graph(vendor: str) -> CompiledStateGraph:
    llm_model = get_chat_model(vendor)

    assign_to_research_agent = create_handoff_tool(agent_name='research_agent', description='Assign task to the research agent.')
    assing_to_calculator_agent = create_handoff_tool(agent_name='calculator_agent', description='Assing task to the calculator agent.')
    assing_to_writer_agent = create_handoff_tool(agent_name='writer_agent', description='Assing task to the writer agent.')

    researcher_agent = create_react_agent(
        name='researcher_agent',
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

    supervisor_agent = create_react_agent(
        name='supervisor',
        model=llm_model,
        tools=[assign_to_research_agent, assing_to_calculator_agent, assing_to_writer_agent],
        prompt=(
            'You are a supervisor managing three agents:\n'
            '- A research agent. Assign research related tasks to this agent.\n'
            '- A calculator agent. Assing math tasks to this agent.\n'
            '- A writer agent. Assing text generation tasks to this agent.\n'
            'Assing work to one agent at a time, do not call agents in parallel.\n'
            'Do not do any work yourself.\n'
            'Never write the responses to the user messages, assign the writer agent to to that, always.'
        )
    )
    #supervisor_agent = llm_model.bind_tools([assign_to_research_agent, assing_to_calculator_agent, assing_to_writer_agent])

    graph = StateGraph(MessagesState)

    graph.add_node(supervisor_agent)
    graph.add_node(researcher_agent)
    graph.add_node(calculator_agent)
    graph.add_node(writer_agent)

    graph.set_entry_point('supervisor')
    graph.add_edge('researcher_agent', 'supervisor')
    graph.add_edge('calculator_agent', 'supervisor')
    graph.add_edge('writer_agent', 'supervisor')

    return graph.compile()


def query_llm(system_graph: CompiledStateGraph, chat_history: ChatHistory) -> None:
    output = system_graph.invoke({'messages': chat_history.messages})
    new_messages = output['messages'][len(chat_history.messages):]
    for m in new_messages:
        if isinstance(m, BaseMessage):
            chat_history.add_message(m)
            if m.content: print(f'{m.content}')

def query_llm_stream(system_graph: CompiledStateGraph, chat_history: ChatHistory) -> None:
    for chunk in system_graph.stream({'messages': chat_history.messages}):
        try:
            new_messages = chunk['supervisor']['messages'][len(chat_history.messages):]
        except:
            new_messages = []

        for m in new_messages:
            if isinstance(m, BaseMessage):
                if m.content: print(m.content, flush=True)
                chat_history.add_message(m)
            elif isinstance(m, dict) and m['role'] == 'tool':
                print(f'{text_colors["violet2"]}Tool: {m["content"]}{text_colors["blue2"]}', flush=True)


def chat(system_graph: CompiledStateGraph, chat_history: ChatHistory, stream: bool) -> None:
    user_input = input(f'\n{text_colors["green2"]}User ("quit" to exit): ')
    print()
    while user_input != 'quit' and user_input != 'exit':
        chat_history.add_message(HumanMessage(content=user_input))

        print(text_colors['blue2'], end='', flush=True)
        query_llm_stream(system_graph, chat_history) if stream else query_llm(system_graph, chat_history)

        user_input = input(f'\n{text_colors["green2"]}User: ')
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

    system_graph = create_system_graph(args.vendor)

    chat(system_graph, chat_history, args.stream)
    chat_history.save_messages()

    conn.close()


main()
