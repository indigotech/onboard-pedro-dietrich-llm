import base64
import pickle
from dotenv import load_dotenv
from enum import Enum
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

from chat_config import *
from chat_history import ChatHistory
from db import init_db


CHAT_WINDOW_SIZE: int = 5


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
            '- A writer agent. Assign text generation tasks to this agent.\n'
            'Assign work to one agent at a time, do not call agents in parallel.\n'
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
            '- A calculator agent. Assign math tasks to this agent.\n'
            'Do not do any math work yourself.'
        )
    ))

    return agents


class ChatbotSystems(Enum):
    RESEARCH = 'research_supervisor'
    MATH = 'calculator_supervisor'

class RouterOutput(BaseModel):
    """
        Router's structured response to decide which multi-agent system will be used to answer the user.
        The `decision` must be either `RESEARCH` or `MATH`. Use `RESEARCH` by default.
    """
    decision: ChatbotSystems
    reason: str = Field(..., description='Why this routing decision was made.')

class ContextOutput(BaseModel):
    """
        Context agent structured output that keeps all the relevant information from the entire chatbot conversation.
        Keep the context data as short as possible, while keeping the important information.
    """
    chat_summary: str = Field(..., description='Summary of the current conversation between the user and the chatbot system.')
    user_data: str = Field(..., description='Relevant information about the user.')

class GraphState(MessagesState):
    context: ContextOutput


def create_router(model: BaseChatModel):
    llm = model.with_structured_output(RouterOutput)

    def router(state: GraphState) -> str:
        last_message = state['messages'][-1]
        result: RouterOutput = llm.invoke([last_message]) # type: ignore
        print(f'{text_colors["cyan2"]}Using {result.decision.name} system.\n')
        return result.decision.value

    return router

def create_context_agent(model: BaseChatModel):
    context_agent = create_react_agent(
        name='context_agent',
        model=model,
        tools=[],
        response_format=ContextOutput,
        prompt=(
            'You are a context agent. You must update the context dictionary with relevant information.\n'
            'The context dictionary must contain relevant user information, useful facts, summaries and other\n'
            'information that should persist across calls to the chatbot.\n'
            'Make the context text as small and brief as possible while keeping the important data.\n'
            'Finish the conversation immediately after generating the context. You must not call any tools.'
        )
    )

    def context_call(state: GraphState) -> GraphState:
        invokeState: GraphState = state
        invokeState['messages'].append(SystemMessage(content=f'Current context of the conversation:\n{state["context"]}'))
        result = context_agent.invoke(invokeState)

        current_context: ContextOutput = result['structured_response']
        state['context'] = current_context

        return state

    return context_call


def build_graph(model: BaseChatModel) -> CompiledStateGraph:
    agents = create_agents(model)
    router = create_router(model)
    context_agent = create_context_agent(model)

    graph = StateGraph(GraphState)

    for agent in agents:
        graph.add_node(agent)
    graph.add_node('context_agent', context_agent)

    graph.add_conditional_edges(START, router)
    graph.add_edge('research_agent', 'research_supervisor')
    graph.add_edge('writer_agent', 'research_supervisor')
    graph.add_edge('calculator_agent', 'calculator_supervisor')
    graph.add_edge('research_supervisor', 'context_agent')
    graph.add_edge('calculator_supervisor', 'context_agent')
    graph.set_finish_point('context_agent')

    return graph.compile()


def encode_context(context: ContextOutput) -> str:
    return base64.b64encode(pickle.dumps(context)).decode('utf-8')

def decode_context(encoding: str) -> ContextOutput:
    if encoding == '': return ContextOutput(chat_summary='', user_data='')
    return pickle.loads(base64.b64decode(encoding.encode('utf-8')))


def query_llm(graph: CompiledStateGraph, chat_history: ChatHistory) -> None:
    last_messages = chat_history.messages[-CHAT_WINDOW_SIZE:]

    output = graph.invoke({'messages': last_messages, 'context': decode_context(chat_history.context)})

    new_messages = output['messages'][len(last_messages):]
    for m in new_messages:
        if isinstance(m, BaseMessage):
            chat_history.add_message(m)
            name = str(m.name)
            if m.content and ('_supervisor' in name or 'transfer_to_' in name or name == 'writer_agent' or name == 'context_agent'):
                print(f'{text_colors["blue2"]}{m.content}\n', flush=True)

    context: ContextOutput = output['context']
    chat_history.update_context(encode_context(context))

def chat(graph: CompiledStateGraph, chat_history: ChatHistory) -> None:
    user_input = input(f'\n{text_colors["green2"]}User ("quit" to exit): ')
    print(flush=True)

    while user_input != 'quit' and user_input != 'exit':
        chat_history.add_message(HumanMessage(content=user_input))

        print(text_colors['blue2'], end='', flush=True)
        query_llm(graph, chat_history)

        user_input = input(f'{text_colors["green2"]}User: ')
        print(flush=True)


def main():
    load_dotenv()
    args = get_arguments()
    conn = init_db()

    chat_history = ChatHistory(args.chatid, conn)

    model = get_chat_model(args.vendor)
    graph = build_graph(model)

    chat(graph, chat_history)
    chat_history.save_messages()
    chat_history.save_context()

    conn.close()


main()
