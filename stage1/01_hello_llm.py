"""
第 1 课：用最裸的方式调用一次 LLM —— Agent 世界的基本粒子
================================================================
学习目标：
  1. 看懂一次 LLM 调用的完整结构：客户端 → messages → 响应对象
  2. 理解 messages 的三种角色（system / user / assistant）
  3. 明白"智谱 GLM 走 OpenAI 兼容端点"意味着什么（可迁移技能）

运行：python 01_hello_llm.py
没有 key 也能跑：自动进入 Mock 模式，先看清代码结构（MT30 同款思路）。
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

# ---- 配置区 -----------------------------------------------------------
# load_dotenv() 从项目根目录的 .env 读取环境变量（key 不写死在代码里）
load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
# 关键点：智谱把自家 API 做成了 OpenAI 兼容格式。
# 换成 DeepSeek/Moonshot/本地 Ollama？只改 base_url，其余代码一行不动。
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash")


# ---- 核心逻辑 ---------------------------------------------------------
def chat(messages: list[dict], temperature: float = 0.7) -> str:
    """一次最小完整的 LLM 调用。

    messages 结构：[{"role": 角色, "content": 文本}, ...]
    temperature：0=几乎确定（适合抽取/分类），1+=更随机（适合创作）
    """
    if API_KEY:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
        )
        # 响应结构：resp.choices[0].message.content —— 先记住，后面解释
        return resp.choices[0].message.content

    # Mock 替身：没有 key 时返回固定结构，保证链路可验证
    last = messages[-1]["content"]
    return f"[Mock回复] 收到 {len(messages)} 条消息，最后一条是：{last!r}"


if __name__ == "__main__":
    if not API_KEY:
        print(">> 未检测到 ZHIPU_API_KEY，进入 Mock 模式（仅验证代码结构）\n")

    # 1. system：设定人格/规则，用户看不见 → "导演的剧本"
    # 2. user：用户的输入 → "演员的台词"
    # 3. assistant：模型的回复 → 模型自己说过的会进入后续上下文
    messages = [
        {"role": "system", "content": "你是一个说话极其简洁的Python导师，每次回答不超过两句话。"},
        {"role": "user", "content": "用一句话解释什么是大语言模型。"},
    ]

    reply = chat(messages, temperature=0.7)
    print("助手:", reply)

    # 对比实验：temperature=0 同样的问题问两次，观察输出是否几乎一致
    print("\n--- temperature=0，同问题连问两次，观察确定性 ---")
    for i in (1, 2):
        ans = chat(
            [{"role": "user", "content": "用一句话解释什么是大语言模型。"}],
            temperature=0,
        )
        print(f"第{i}次: {ans}")
