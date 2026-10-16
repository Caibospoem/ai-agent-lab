"""
第 4 课（0）：不用任何框架，手写一个最小 RAG —— 把"手写过"补成真的
================================================================
背景：你简历里的 MT30/合规平台是 AI 搭的，0.33 阈值也是 AI 调的——没关系，
这课就是让你把这件事变成"自己的"。这个脚本不依赖 LangChain，
余弦相似度、排序、阈值过滤全部手写，总共不到 80 行核心代码。

RAG 的全部原理，其实就三步：
    ① 入库：每条知识 → 转成向量（一串数字）→ 存起来
    ② 检索：问题 → 转成向量 → 和库里每个向量算余弦相似度 → 取最高的几条
    ③ 生成：把命中的知识塞进 prompt → 问模型 → 模型照着答

什么是余弦相似度？两个向量夹角的"接近程度"，1=方向完全相同，0=无关。
    手算例子：a=(1,0,1)  b=(1,1,0)
      点积 = 1*1+0*1+1*0 = 1
      |a| = √2，|b| = √2
      cos = 1/(√2*√2) = 0.5   ← 就这么点数学，没了

那 0.33 阈值是干嘛的？（MT30 项目里 AI 给你调的那个数）
    分数 = "这条知识和你问题有多相关"。低于阈值的片段 = 无关噪音，
    塞进 prompt 只会让模型被带偏。所以检索完先过滤：全都不达标 → 直接拒答。
    本脚本 THRESHOLD=0.33，跑一次 demo 你就看见它的作用了。

运行：python 00_rag_from_scratch.py    （无 key 也能跑：Mock 向量演示到检索为止）
"""
import hashlib
import json
import math
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
LLM_MODEL = "glm-4-flash"
EMBED_MODEL = "embedding-3"
THRESHOLD = 0.33  # MT30 同款起点：低于它的检索结果视为"没检索到"

# ---- 知识库：8 句话，就是 8 个字符串，没了 ------------------------------
KB = [
    "本训练营课程核心公式：Agent = LLM + 工具 + 循环。",
    "训练营收取学员的门槛是：完成上一课作业并通过思考题考核。",
    "训练营阶段二主题：RAG 与记忆策略，共 3 课（第4-6课）。",
    "训练营阶段三主题：LangGraph 状态机与多 Agent 协作，共 4 课。",
    "GLM-4-Flash 是智谱的免费对话模型，支持 Function Calling。",
    "智谱 embedding-3 是文本向量化模型，用于语义检索。",
    "wttr.in 是免费无 key 的天气 API，第 3 课用作 Agent 工具。",
    "MCP 是 Anthropic 提出的工具接入协议，比喻为 Agent 的 USB-C。",
    "作者刘曾做过的项目有 MT30合规平台、RAG、基于AnimeGANv3的图片风格迁移微信小程序-风韵转换、人脸识别系统、智能小夜灯-派蒙、“星露谷”邻里互助平台。",

]

# ---- 第 1 步：向量化 —— 把文字变成一串数字 -----------------------------
def embed_texts(texts: list[str]) -> list[list[float]]:
    if API_KEY:  # 真模式：embedding-3，每条文本变成 2048 维浮点向量
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in resp.data]

    # Mock：按字符哈希的简易向量（同字越多越相似，仅求能跑通，别当真语义）
    dim = 64
    vecs = []
    for t in texts:
        v = [0.0] * dim
        for ch in set(t):
            h = int(hashlib.md5(ch.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
        vecs.append(v)
    return vecs


# ---- 第 2 步：余弦相似度 —— 手写，就这 6 行 ----------------------------
def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def retrieve(question: str, kb_vecs: list[list[float]], top_k: int = 3):
    """检索 = 对库里的每条知识算一遍相似度，排序取前 k 条"""
    q_vec = embed_texts([question])[0]
    scores = [(cosine(q_vec, kv), text) for kv, text in zip(kb_vecs, KB)]
    scores.sort(key=lambda x: -x[0])
    return scores[:top_k]


# ---- 第 3 步：生成 —— 把命中的知识塞进 prompt 问模型 -------------------
def generate(question: str, contexts: list[str]) -> str:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    context_block = "\n".join(f"- {c}" for c in contexts)
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": (
                "只根据参考知识回答，答案末尾注明引用来源；参考知识不足以回答时，"
                "直接回复\"知识库中没有相关内容\"，禁止编造。\n\n"
                f"参考知识：\n{context_block}"
            )},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


def answer(question: str) -> None:
    kb_vecs = embed_texts(KB)  # 生产中入库时算好存起来，这里每次重算图省事
    hits = retrieve(question, kb_vecs)

    print(f"问: {question}")
    print("  [检索] 全库相似度排名:")
    for score, text in hits:
        flag = "✓" if score >= THRESHOLD else f"✗ 低于{THRESHOLD}，过滤"
        print(f"    {score:.3f} {flag}  {text[:30]}…")

    valid = [text for score, text in hits if score >= THRESHOLD]
    if not valid:
        print("答: [阈值拦截] 最高分都没过线 → 拒答，根本不调 LLM（省钱的防线）\n")
        return
    if not API_KEY:
        print(f"答: [无key·演示到检索为止] 将把这 {len(valid)} 条塞进 prompt 问 {LLM_MODEL}\n")
        return
    print(f"答: {generate(question, valid)}\n")


if __name__ == "__main__":
    print("=" * 64)
    print("实验一：库内问题 —— 看正确知识如何浮到顶部并通过阈值")
    print("=" * 64)
    answer("训练营阶段二学什么？")

    print("=" * 64)
    print("实验二：库外问题 —— 看分数全部低空坠落，阈值拦截拒答")
    print("=" * 64)
    answer("今天A股大盘涨了几个点？")

    print("=" * 64)
    print("实验三：自定义问题 —— 看检索结果如何被塞进 prompt 让模型回答")
    print("=" * 64)
    answer("列举一些作者刘曾做过的项目")
