import argparse
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vendor', type=str, choices=['openai', 'groq'], default='openai')
    parser.add_argument('-s', '--stream', action='store_true', default=False)
    return parser.parse_args()

def get_chat_model(vendor: str):
    print(f'Selected vendor: {vendor}')

    if not os.environ.get(f'{vendor.upper()}_API_KEY'):
        raise ValueError(f'API key not defined for the vendor {vendor}.')

    if vendor == 'openai':
        return init_chat_model('gpt-4o-mini', model_provider=vendor)
    else:
        return init_chat_model('llama-3.3-70b-versatile', model_provider=vendor)

def chat(model: BaseChatModel, stream: bool):
    chat_history: list[BaseMessage] = [
        SystemMessage(content='You are a helpful assistant, but you always use extremely elaborate language in your responses.')
    ]

    user_input = input('Prompt ("quit" to exit): ')
    while user_input != 'quit':
        print()
        chat_history.append(HumanMessage(content=user_input))

        ai_response: str = ''
        if stream:
            for chunk in model.stream(chat_history):
                print(chunk.content, end='', flush=True)
                ai_response += str(chunk.content)
            print('\n')
        else:
            output_message = model.invoke(chat_history)
            print(output_message.content, '\n')
            ai_response = str(output_message.content)

        chat_history.append(AIMessage(content=ai_response))

        user_input = input('Prompt ("quit" to exit): ')


def main():
    load_dotenv()
    args = get_arguments()

    model = get_chat_model(args.vendor)
    if args.stream:
        print('Using streaming mode.\n')
    else:
        print('Streaming mode disabled.\n')

    chat(model, args.stream)


main()
