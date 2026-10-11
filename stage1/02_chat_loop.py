"""
第 2 课：多轮对话循环 —— 你人生第一个有"记忆"的程序
================================================================
本课核心认知（面试高频）：
  LLM 本身是无状态的！服务器不保存任何对话记录。
  所谓"记忆"，是客户端把【全部历史消息】每次都完整重发一遍。
  所以第 2 课 = 第 1 课 + 一个不断变长的 messages 列表。

  上下文窗口（context window）限制的就是这个列表的最大长度，
  超长就要截断/摘要 —— 这正是第 5 课"记忆策略"的伏笔。

感知-决策-行动循环（Agent 的基本节拍）在这里首次出现：
  感知 = 用户输入追加进 messages
  决策+行动 = 带全量历史调用模型，得到回复
  记录 = 回复也追加进 messages（漏了这步，下一轮模型就"失忆"）

运行：python 02_chat_loop.py
命令：输入 q 退出；输入 reset 清空记忆（观察与不 reset 的区别）
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash")

SYSTEM_PROMPT = "你是终端助手'小智'，回答保持在两句话以内。"


def chat(messages: list[dict], temperature: float = 0.7) -> str:
    if API_KEY:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=temperature
        )
        return resp.choices[0].message.content

    # Mock 替身：故意把"历史在变长"打印出来，让无状态→有记忆的机制可视化
    user_turns = sum(1 for m in messages if m["role"] == "user")
    return (
        f"[Mock回复] 第 {user_turns} 轮。本次请求携带 {len(messages)} 条历史消息。"
        f"（真实 key 下我会认真回答；注意消息数在增长）"
    )


def main() -> None:
    if not API_KEY:
        print(">> Mock 模式：填入真实 key 重跑可体验真对话\n")

    print(f"正在与 {MODEL} 对话 | q=退出 | reset=清空记忆\n")

    # 记忆的载体就是这一个列表；system 消息永远待在队首
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() == "q":
            print("再见！")
            break
        if user_input.lower() == "reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print(">> 记忆已清空（messages 列表被重置）\n")
            continue
        if user_input.lower() == "history": # history 命令
            print(">> 当前历史消息：")
            print(json.dumps(messages, ensure_ascii=False, indent=2))
            print()
            continue

        # [感知] 用户输入进入历史
        messages.append({"role": "user", "content": user_input})
        # [决策+行动] 带"全部历史"去请求模型
        reply = chat(messages)
        # [记录] 模型的话也进入历史 —— 没有这一行就没有连续对话
        messages.append({"role": "assistant", "content": reply})
        # 同时保留系统消息(system)和最近20条消息(10轮对话)
        if len(messages) > 21:
            system_message = messages[0]
            recent_messages = messages[-20:]
            messages = [system_message] + recent_messages
            print(f">> 历史消息过长，已截断，保留最近20条对话 (共{len(messages)}条)\n")
        # 更新最新的20条对话

        print(f"小智: {reply}    [当前历史 {len(messages)} 条]\n")


if __name__ == "__main__":
    main()
