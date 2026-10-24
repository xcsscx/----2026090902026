import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

# 读取项目根目录下的 .env（不存在时不会报错）
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")

SYSTEM_PROMPT = "你是一个乐于助人的助手，请用中文回答。"


def ask(client, question):
    """向模型发送一条消息，返回回答文本。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content


def main():
    if not API_KEY:
        sys.exit(
            "未检测到 API_KEY。\n"
            "请在项目根目录创建 .env 文件（可参考 .env.example），写入：\n"
            "  API_KEY=你的密钥\n"
            "  BASE_URL=https://api.deepseek.com\n"
            "  MODEL=deepseek-chat"
        )

    # 兼容 OpenAI 接口协议的客户端，换平台只需改 .env 中的 BASE_URL 和 MODEL
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"模型：{MODEL}")
    print(f"接口地址：{BASE_URL}")
    print("输入问题后回车即可提问，输入 q 退出。")

    while True:
        try:
            question = input("\n请输入你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if question.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            break
        if not question:
            print("问题不能为空，请重新输入。")
            continue

        print("正在请求模型……")
        try:
            answer = ask(client, question)
        except Exception as exc:
            print(f"API 调用失败：{exc}")
            continue

        print("\n模型回答：")
        print(answer)


if __name__ == "__main__":
    main()
