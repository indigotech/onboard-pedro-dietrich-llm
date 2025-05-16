from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from chat_config import get_arguments, get_chat_model


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
