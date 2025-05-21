from dotenv import load_dotenv
from langchain_core.tools import InjectedToolCallId
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing import Annotated, Literal

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
            'content': f'{text_colors["violet2"]}Successfully transfered to {agent_name}.',
            'name': name,
            'tool_call_id': tool_call_id
        }
        return Command(goto=agent_name, update={**state, 'messages': state['messages'] + [tool_message]}, graph=Command.PARENT)

    return handoff_tool


def create_agents(model: BaseChatModel) -> list[CompiledGraph]:
    agents: list[CompiledGraph] = []

    assign_to_research_agent = create_handoff_tool(agent_name='research_agent', description='Assign task to the research agent.')
    assign_to_calculator_agent = create_handoff_tool(agent_name='calculator_agent', description='Assign task to the calculator agent.')
    assign_to_writer_agent = create_handoff_tool(agent_name='writer_agent', description='Assign task to the writer agent.')

    agents.append(create_react_agent(
        name='research_agent',
        model=model,
        tools=[web_search],
        prompt=(
            'You are a research agent. You only perform web research tasks, using the web_search tool.\n'
            'As soon as you finish the search, you return to your supervisor the search results.\n'
            'Do not include extra text in the search results.'
        )
    ))
    agents.append(create_react_agent(
        name='calculator_agent',
        model=model,
        tools=[add, subtract, multiply],
        prompt=(
            'You are a calculator agent. Do only sum, subtraction and multiplication, and nothing else.\n'
            'Respond directly to your supervisor, and do not include any text other than the task results.'
        )
    ))
    agents.append(create_react_agent(
        name='writer_agent',
        model=model,
        tools=[],
        prompt=(
            'You are a writer agent. You should only generate text for the response.\n'
            'Respond directly to your supervisor.'
        )
    ))
    agents.append(create_react_agent(
        name='research_supervisor',
        model=model,
        tools=[assign_to_research_agent, assign_to_writer_agent],
        prompt=(
            'You are a supervisor managing two agents:\n'
            '- A research agent. Assign research related tasks to this agent.\n'
            '- A writer agent. Assing text generation tasks to this agent.\n'
            'Assing work to one agent at a time, do not call agents in parallel.\n'
            'Do not do any work yourself.\n'
            'Never write the responses to the user messages, assign the writer agent to do that, always.'
        )
    ))
    agents.append(create_react_agent(
        name='calculator_supervisor',
        model=model,
        tools=[assign_to_calculator_agent],
        prompt=(
            'You are a supervisor managing one agent:\n'
            '- A calculator agent. Assing math tasks to this agent.\n'
            'Do not do any math work yourself.'
        )
    ))

    return agents


class GraphState(MessagesState):
    route: str

class RouterOutput(BaseModel):
    """
        Router's structured response to decide which multi-agent system will be used to answer the user.
        The `decision` must be either `research` or `math`. Use `research` by default.
    """
    decision: Literal['research', 'math']
    reason: str = Field(..., description='Why this routing decision was made.')


def create_router(model: BaseChatModel):
    llm = model.with_structured_output(RouterOutput)

    def router(state: GraphState) -> Literal['research', 'math']:
        latest_message = state['messages'][-1]
        result: RouterOutput = llm.invoke([latest_message]) # type: ignore
        print(f'{text_colors["cyan2"]}Using {result.decision} system.\n')
        return result.decision

    return router


def build_graph(model: BaseChatModel) -> CompiledStateGraph:
    agents = create_agents(model)
    router = create_router(model)

    graph = StateGraph(GraphState)

    for agent in agents:
        graph.add_node(agent)

    graph.add_conditional_edges(START, router, {'research': 'research_supervisor', 'math': 'calculator_supervisor'})
    graph.add_edge('research_agent', 'research_supervisor')
    graph.add_edge('writer_agent', 'research_supervisor')
    graph.add_edge('calculator_agent', 'calculator_supervisor')

    return graph.compile()


def query_llm(graph: CompiledStateGraph, chat_history: ChatHistory) -> None:
    output = graph.invoke({'messages': chat_history.messages})
    new_messages = output['messages'][len(chat_history.messages):]
    for m in new_messages:
        if isinstance(m, BaseMessage):
            chat_history.add_message(m)
            name = str(m.name)
            if m.content and ('_supervisor' in name or 'transfer_to_' in name or name == 'writer_agent'):
                print(f'{text_colors["blue2"]}{m.content}\n', flush=True)

def query_llm_stream(graph: CompiledStateGraph, chat_history: ChatHistory) -> None:
    for chunk in graph.stream({'messages': chat_history.messages}):
        new_messages: list[BaseMessage] = chunk.get('research_supervisor', {}).get('messages', [])[len(chat_history.messages):]
        new_messages.extend(chunk.get('calculator_supervisor', {}).get('messages', [])[len(chat_history.messages):])
        new_messages.extend(chunk.get('writer_agent', {}).get('messages', [])[len(chat_history.messages):])

        for m in new_messages:
            if isinstance(m, BaseMessage):
                if m.content: print(f'{text_colors["blue2"]}{m.content}', flush=True)
                chat_history.add_message(m)
            elif isinstance(m, dict) and m['role'] == 'tool':
                print(f'{text_colors["violet2"]}Tool: {m["content"]}\n', flush=True)
    print(flush=True)

def chat(graph: CompiledStateGraph, chat_history: ChatHistory, stream: bool) -> None:
    user_input = input(f'\n{text_colors["green2"]}User ("quit" to exit): ')
    print(flush=True)

    while user_input != 'quit' and user_input != 'exit':
        chat_history.add_message(HumanMessage(content=user_input))

        print(text_colors['blue2'], end='', flush=True)
        query_llm_stream(graph, chat_history) if stream else query_llm(graph, chat_history)

        user_input = input(f'{text_colors["green2"]}User: ')
        print(flush=True)


def main():
    load_dotenv()
    args = get_arguments()
    conn = init_db()

    if args.stream:
        print('Using streaming mode.\n')
    else:
        print('Streaming mode disabled.\n')

    chat_history = ChatHistory(args.chatid, conn)

    model = get_chat_model(args.vendor)
    graph = build_graph(model)

    chat(graph, chat_history, args.stream)
    chat_history.save_messages()

    conn.close()


main()
