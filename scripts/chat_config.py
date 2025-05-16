import argparse
import os
from datetime import datetime
from typing import NamedTuple
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

class MessageData(NamedTuple):
    time: datetime
    role: str
    content: str

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vendor', type=str, choices=['openai', 'groq'], default='openai')
    parser.add_argument('-s', '--stream', action='store_true', default=False)
    parser.add_argument('-c', '--chatid', type=str)
    return parser.parse_args()

def get_chat_model(vendor: str) -> BaseChatModel:
    print(f'Selected vendor: {vendor}')

    if not os.environ.get(f'{vendor.upper()}_API_KEY'):
        raise ValueError(f'API key not defined for the vendor {vendor}.')

    if vendor == 'openai':
        return init_chat_model('gpt-4o-mini', model_provider=vendor)
    else:
        return init_chat_model('llama-3.3-70b-versatile', model_provider=vendor)

def get_message_role(message: BaseMessage) -> str:
    if isinstance(message, SystemMessage):
        return 'system'
    elif isinstance(message, HumanMessage):
        return 'user'
    elif isinstance(message, AIMessage):
        return 'assistant'
    elif isinstance(message, ToolMessage):
        return 'tool'
    else:
        raise ValueError(f'Unsupported message type: {type(message)}')


text_colors = {
    'normal': '\33[0m',
    'black': '\33[30m',
    'red': '\33[31m',
    'green': '\33[32m',
    'yellow': '\33[33m',
    'blue': '\33[34m',
    'violet': '\33[35m',
    'beige': '\33[36m',
    'white': '\33[37m',
    'gray': '\33[90m',
    'red2': '\33[91m',
    'green2': '\33[92m',
    'yellow2': '\33[93m',
    'blue2': '\33[94m',
    'violet2': '\33[95m',
    'beige2': '\33[96m',
    'white2': '\33[97m',
}
