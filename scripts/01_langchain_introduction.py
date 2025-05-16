from dotenv import load_dotenv

from chat_config import get_arguments, get_chat_model

def main():
    load_dotenv()
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


main()
