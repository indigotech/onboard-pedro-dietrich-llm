import argparse
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vendor', type=str, choices=['openai', 'groq'], default='openai')
    parser.add_argument('-s', '--stream', action='store_true', default=False)
    return parser.parse_args()

def get_chat_model(vendor: str):
    print(f'Selected vendor: {vendor}')

    if not os.environ.get(f'{vendor.upper()}_API_KEY'):
        print(f'API key not defined for the vendor {vendor}.\n')

    if vendor == 'openai':
        return init_chat_model('gpt-4o-mini', model_provider=vendor)
    elif vendor == 'groq':
        return init_chat_model('llama-3.3-70b-versatile', model_provider=vendor)
    else:
        raise ValueError(f"Unsupported vendor: {vendor}.")


def main():
    args = get_arguments()
    model = get_chat_model(args.vendor)

    prompt = input('Prompt: ')

    if args.stream:
        print('Using streaming mode.\n')
        for chunk in model.stream(prompt):
            print(chunk.content, end='', flush=True)
        print()
    else:
        print('Streaming mode disabled.\n')
        message = model.invoke(prompt)
        print(message.content)


load_dotenv()
main()
