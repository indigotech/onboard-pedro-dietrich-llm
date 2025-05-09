import argparse
import os
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

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
