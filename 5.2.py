import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")

SYSTEM_PROMPT = """你是一个校园信息助手。请阅读用户提供的校园相关文本，并严格按下面的固定格式输出，不要添加任何额外说明或标题：

一句话总结：<用一句话概括这段文本或问题>
关键信息：<提取时间、地点、人物、课程、作业等信息，用分号分隔；如果用户是在提问，就在这里给出简要回答>
待办事项：<需要用户执行的事项；如果没有，写“无”>"""


def ask(client, text):
    """把校园文本交给模型，返回固定格式的结果。"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def run_one(client, text):
    """处理一条文本：请求模型并打印结果。"""
    print(f"\n输入：{text}")
    print("正在请求模型……")
    try:
        answer = ask(client, text)
    except Exception as exc:
        print(f"API 调用失败：{exc}")
        return
    print("\n输出：")
    print(answer)


DEMO_TEXTS = [
    "本周五前提交高数作业，作业包括第三章习题 1、3、5，提交到学习通。",
    "周六下午三点将在图书馆报告厅举办人工智能讲座，主讲人是计算机学院李教授，请提前十分钟入场。",
    "Python 中列表和元组有什么区别？什么时候应该使用元组？",
]


def main():
    if not API_KEY:
        sys.exit(
            "未检测到 API_KEY。\n"
            "请在项目根目录的 .env 文件中配置 API_KEY、BASE_URL、MODEL。"
        )

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    if "--demo" in sys.argv:
        # 自动运行三类示例
        for text in DEMO_TEXTS:
            run_one(client, text)
        return

    print(f"模型：{MODEL}")
    print("请输入你的问题或校园文本，回车提交（输入 q 退出）：")

    while True:
        try:
            text = input("\n请输入：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if text.lower() in {"q", "quit", "exit"}:
            print("已退出。")
            break
        if not text:
            print("输入不能为空，请重新输入。")
            continue

        run_one(client, text)
        print("\n" + "-" * 40)


if __name__ == "__main__":
    main()
