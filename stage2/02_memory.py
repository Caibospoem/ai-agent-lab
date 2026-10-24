"""
第 5 课：对话记忆策略 —— 从第 2 课的"手动截断"到三层记忆
================================================================
第 2 课你写过 messages[-20:] 截断（短期记忆的雏形），第 4 课学了 RAG。
这课把两者拼成产品级的记忆架构——ChatGPT 的 memory、Claude 的记忆功能，
内核都是这个结构：

    ┌ 短期记忆：最近 N 条原文，一字不动（当前对话的"工作记忆"）
    ├ 中期记忆：滚动摘要（LLM 把旧对话压成一段 gist，保留人名/数字/承诺）
    └ 长期记忆：历史对话向量化入库，每轮按当前问题检索相关记忆注入
       ↑ 这就是"对话历史上的 RAG"——第 4 课的技能直接复用

为什么三层缺一不可：
    只截断 → 早期事实直接消失（"我叫什么"忘了）
    只摘要 → 细节被压扁（摘要说"讨论过项目"，但 0.33 这个数字没了）
    只向量 → 每轮都检索，重复啰嗦且成本高
    组合   → 近期靠原文、中期靠摘要、细节靠检索，各管一段

运行：python 02_memory.py        （同一段对话跑两遍：截断模式 vs 三层记忆模式）
     python 02_memory.py chat   （交互模式，自己聊）
无 key 也能跑：Mock 模式下 LLM 回复是假的，但【记忆注入=有/无】的结构对比是真的。
"""
import hashlib
import math
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

import re

# 显式记忆触发：匹配 "记住：xxx" / "记住:xxx"（兼容中英文冒号，允许前面有"请帮我"等）
REMEMBER_RE = re.compile(r"记住[：:]\s*(.+)")


load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
LLM_MODEL = "glm-4-flash"
EMBED_MODEL = "embedding-3"

MAX_RECENT = 4       # 短期记忆：摘要后保留最近 4 条消息（2 轮）
SUMMARY_TRIGGER = 8  # messages 超过 8 条 → 触发滚动摘要
TOP_K = 2            # 长期记忆：每轮检索相关历史条数

SYSTEM = "你是助手'小忆'，回答简洁。如果提供了长期记忆或对话摘要，可以自然引用它们。"


# ---- 向量化与相似度：第 4 课手写版直接复用 ------------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    if API_KEY:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in resp.data]
    dim = 64  # Mock：字符哈希向量（同字越多越相似，能演示结构即可）
    vecs = []
    for t in texts:
        v = [0.0] * dim
        for ch in set(t):
            h = int(hashlib.md5(ch.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
        vecs.append(v)
    return vecs


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ---- LLM 调用 -----------------------------------------------------------
def chat_llm(messages, temperature=0.3):
    if API_KEY:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        return client.chat.completions.create(
            model=LLM_MODEL, messages=messages, temperature=temperature,
        ).choices[0].message.content
    sys0 = messages[0]["content"]
    has_mem = ("长期记忆" in sys0) or ("对话摘要" in sys0)
    last = messages[-1]["content"]
    return f"[Mock·记忆注入={'有' if has_mem else '无'}] 收到：{last[:24]}"


def summarize_llm(old_summary: str, dialogue: str) -> str:
    prompt = (
        "把【旧摘要】和【新对话】合并成不超过120字的摘要。"
        "必须保留：人名、项目名、数字、日期、承诺。直接输出摘要。\n"
        f"【旧摘要】{old_summary or '（无）'}\n【新对话】\n{dialogue}"
    )
    if API_KEY:
        return chat_llm([{"role": "user", "content": prompt}], temperature=0.1)
    return f"[Mock摘要] 之前聊过：{dialogue[:40]}…"


# ---- 三层记忆 Agent ------------------------------------------------------
class MemoryBot:
    def __init__(self, use_memory: bool = True):
        self.use_memory = use_memory
        self.summary = ""
        self.messages = []
        self.long_term = []
        self.round = 0

    def recall(self, query: str):
        """长期记忆 = 对话历史上的 RAG（不变）"""
        if not self.use_memory or not self.long_term:
            return []
        qv = embed_texts([query])[0]
        scored = sorted(
            ((cosine(qv, v), txt, r) for v, txt, r in self.long_term),
            key=lambda x: -x[0])
        return scored[:TOP_K]

    def _compress(self):
        """中期记忆（不变）"""
        old, keep = self.messages[:-MAX_RECENT], self.messages[-MAX_RECENT:]
        dialogue = "\n".join(f"{r}: {c}" for r, c in old)
        self.summary = summarize_llm(self.summary, dialogue)
        self.messages = keep

    def _write_memory(self, fact: str):
        """显式写入长期记忆：向量用 fact 原文（保证语义检索命中），展示文本带溯源"""
        self.long_term.append(
            (embed_texts([fact])[0], f"第{self.round}轮 用户要求记住: {fact}", self.round)
        )

    def chat(self, user_input: str):
        self.round += 1
        mems = self.recall(user_input)

        # 组装 prompt（同原来）
        parts = [SYSTEM]
        if mems:
            parts.append("【长期记忆·相关历史】\n" + "\n".join(f"- {txt}" for _, txt, _ in mems))
        if self.summary:
            parts.append("【对话摘要】\n" + self.summary)
        msgs = [{"role": "system", "content": "\n\n".join(parts)}]
        msgs += [{"role": r, "content": c} for r, c in self.messages]
        msgs.append({"role": "user", "content": user_input})

        # ★ 核心改动：默认不走 LLM 后不入库；只有显式"记住："才写长期记忆
        m = REMEMBER_RE.search(user_input)
        if m:
            fact = m.group(1).strip()
            self._write_memory(fact)
            reply = f"好的，我记住了：{fact}"     # 程序化确认，不依赖 LLM
        else:
            reply = chat_llm(msgs)

        # 短期记忆照常更新（user 原话仍进窗口，20轮后自然被截断/压缩）
        self.messages += [("user", user_input), ("assistant", reply)]
        if len(self.messages) > SUMMARY_TRIGGER:
            self._compress()

        mem_str = f"长期记忆{len(mems)}条" + (f"(top:{mems[0][0]:.2f})" if mems else "")
        state = f"[短期{len(self.messages)}条 | 摘要{'√' if self.summary else '×'} | 库内{len(self.long_term)}条 | {mem_str}]"
        return reply, state



# ---- 演示：同一段对话，两种记忆模式 --------------------------------------
def run_demo(bot: MemoryBot):
    script = [
        "你好，我叫刘曾，正在准备AI应用岗的面试",
        "我最核心的简历项目是智能合规审查平台，用RAG做的",
        "RAG检索用的embedding-3，相似度阈值0.33",
        "我周末喜欢煲汤，我家的猫叫派蒙",
        "我下周三要去厦门出差",
        "帮我算一下123加456等于多少",
        "好的，那我先去改简历了，回头聊",
    ]
    print(f"\n--- 开始正文对话（{bot.__class__.__name__}）---")
    for turn in script:
        reply, state = bot.chat(turn)
        print(f"你: {turn}\n小忆: {reply}  {state}")

    print("\n--- 检验：这些事发生在早期，早被截断丢出短期窗口 ---")
    for q in ["这次面试准备的核心项目是什么？检索阈值调的多少？",
              "我下周要去哪出差来着？",
              "我叫什么名字？"]:
        reply, state = bot.chat(q)
        print(f"你: {q}\n小忆: {reply}\n    {state}")

FILLER = [
    "帮我算一下 123 乘 456", "我在准备 AI 应用岗面试", "我最近在学 LangChain",
    "讲讲滚动摘要的原理", "我下周三去厦门出差", "帮我算一下 (3.7+1.2)*15",
    "晚上吃什么好", "讲讲向量数据库", "帮我算一下 999 乘 888",
    "推荐几本技术书", "我在用 glm-4-flash", "帮我算一下 2 的 10 次方",
    "讲讲 Turbopack", "RAG 阈值调多少合适", "帮我算一下 7 的 8 次方",
    "明天继续学", "讲讲 Function Calling", "帮我算一下 50 减 17",
    "讲讲 Turbopack 缓存",
]

def run_explicit_test(bot: MemoryBot):
    print("\n--- 第1轮：显式记忆 ---")
    reply, state = bot.chat("记住：我养了只乌龟叫皮皮")
    print(f"你: 记住：我养了只乌龟叫皮皮\n小忆: {reply} {state}")

    print("\n--- 第2~20轮：无关闲聊（把记忆挤出短期窗口）---")
    for t in FILLER:
        reply, state = bot.chat(t)
        print(f"你: {t}\n小忆: {reply} {state}")

    print("\n--- 检验 ---")
    for q in ["我的宠物叫什么？",        # 说过的 → 库里有 → 应回答皮皮
              "我养了什么动物来着？",    # 同义改写 → 语义检索应命中
              "我的猫叫什么名字？"]:     # 没说过的 → 库里没有 → 不应编造
        reply, state = bot.chat(q)
        print(f"你: {q}\n小忆: {reply}\n   {state}")



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        bot = MemoryBot(use_memory=True)
        print("三层记忆模式已就绪 | q 退出 | 试试先聊几轮再问早期的事")
        while True:
            try:
                q = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("q", "quit", "exit") or not q:
                break
            reply, state = bot.chat(q)
            print(f"小忆: {reply}  {state}\n")
    else:
        # print("=" * 60)
        # print("模式一：纯截断（=第2课的做法，只保留最近几条）")
        # print("=" * 60)
        # run_demo(MemoryBot(use_memory=False))
        # print("\n" + "=" * 60)
        # print("模式二：三层记忆（短期截断 + 滚动摘要 + 长期向量检索）")
        # print("=" * 60)
        # run_demo(MemoryBot(use_memory=True))
        # print("\n结论自己看：三个检验问题，哪种模式答得上？")

        print("=" * 60)
        print("显式记忆模式：只有'记住：xxx'才写入长期记忆")
        print("=" * 60)
        run_explicit_test(MemoryBot(use_memory=True))