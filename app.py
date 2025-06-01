from chatbot.chatbot_core import run_chatbot_pipeline

def main():
    while True:
        user_input = input("질문을 입력하세요 (종료: exit): ")
        if user_input.lower() == "exit":
            break
        answer = run_chatbot_pipeline(user_input)
        print(f"\n[응답]\n{answer}\n")

if __name__ == "__main__":
    main()