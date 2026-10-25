"""
第 6 课：混合检索 + RAG 评测 —— 阶段二收官
================================================================
第 4 课 A 股实验你已经见过向量检索的第一种死法（语义无关全灭）。
这课补上第二种死法和解法，再教你怎么用数字证明检索变好了。

向量检索的两种死法：
    ① 用词不同、语义相同（"连不上数据库" vs "数据库连接超时"）→ 语义向量能救
    ② 精确匹配型（错误码 E-4021、工单号、人名）→ 语义向量经常救不了：
       "E-4021"和"报错排查"在向量空间里毫无关系，但关键词一抓一个准
解法 = 混合检索：向量一路 + 关键词一路，RRF 融合排名。

RRF（Reciprocal Rank Fusion）：每路的排名贡献 1/(60+rank)，相加。
    为什么不直接把两路的分数加权平均？——量纲不可比（余弦 0~1，BM25 无上界），
    排名融合天然免归一化，对分数分布鲁棒。

评测三件套（没有评测的 RAG 调优 = 玄学）：
    标注集：几条 query → 哪条知识该被召回（人工标，几十条就够起步）
    Recall@k：该召回的进没进前 k（漏没漏）
    MRR：正确答案排第几（1/名次，越靠前越高）

运行：python 03_hybrid_retrieval.py
输出：6 条 query × 3 种检索的金标排名 + 汇总指标表，看哪路赢在哪类 query 上。
"""
import hashlib
import math
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
EMBED_MODEL = "embedding-3"

# ---- 知识库：故意混入 3 条"精确匹配型"条目 -------------------------------
KB = [
    "训练营课程核心公式：Agent = LLM + 工具 + 循环。",
    "训练营收取学员的门槛：完成上一课作业并通过思考题考核。",
    "训练营阶段二主题：RAG 与记忆策略，共 3 课。",
    "GLM-4-Flash 是智谱的免费对话模型，支持 Function Calling。",
    "智谱 embedding-3 是文本向量化模型，用于语义检索。",
    "错误码 E-4021 表示数据库连接超时，先检查连接池配置。",          # 精确型
    "工单 GD-2026-1103 由王五处理，当前状态：待复现。",              # 精确型
    "省钱技巧：优先用 glm-4-flash 等免费模型，付费前先看免费额度。",  # 语义型
    "wttr.in 是免费无 key 的天气 API。",
    "MCP 是 Anthropic 提出的工具接入协议，比喻为 Agent 的 USB-C。",
]

# 标注集：(query, 金标关键词) —— 金标 = 该被召回的条目里必须出现的字样
GOLD = [
    ("数据库连不上报什么错误", "E-4021"),        # 语义改写型：期望向量路赢
    ("E-4021是什么错误", "E-4021"),              # 精确匹配型：期望关键词路赢
    ("怎么少花点钱", "省钱技巧"),                # 语义改写型
    ("GD-2026-1103 的处理进度", "GD-2026-1103"), # 精确匹配型
    ("embedding-3 是干嘛用的", "embedding-3"),   # 精确偏语义
    ("怎样才能把费用降下来", "省钱技巧"),        # 语义改写型（换个问法再测一次）
]


# ---- 第 4 课手写向量检索，直接复用 ---------------------------------------
def embed_texts(texts):
    if API_KEY:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [item.embedding for item in resp.data]
    dim = 64  # Mock：字符哈希（同字越多越相似）
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


def vector_rank(query, kb_vecs):
    qv = embed_texts([query])[0]
    scored = sorted(((cosine(qv, v), i) for i, v in enumerate(kb_vecs)), key=lambda x: -x[0])
    return [i for _, i in scored]


# ---- 关键词路：字符 bigram 重叠（简化版 BM25 思想：命中越多越相关）--------
def bigrams(text):
    return {text[i:i + 2] for i in range(len(text) - 1)} | set(text)


def keyword_rank(query):
    qb = bigrams(query)
    scores = [(len(qb & bigrams(doc)) / max(len(qb), 1), i) for i, doc in enumerate(KB)]
    scores.sort(key=lambda x: -x[0])
    return [i for _, i in scores]


# ---- RRF 融合：只看排名，不看分数 ----------------------------------------
def rrf_rank(query, kb_vecs, k=60):
    vr, kr = vector_rank(query, kb_vecs), keyword_rank(query)
    fused = {}
    for rank, i in enumerate(vr):
        fused[i] = fused.get(i, 0) + 1 / (k + rank + 1)
    for rank, i in enumerate(kr):
        fused[i] = fused.get(i, 0) + 1 / (k + rank + 1)
    return [i for i, _ in sorted(fused.items(), key=lambda x: -x[1])]


# ---- 评测：Recall@3 与 MRR ------------------------------------------------
def gold_index(gold_key):
    return next(i for i, d in enumerate(KB) if gold_key in d)


def evaluate(rank_fn, name, kb_vecs=None):
    recalls, mrrs, detail = [], [], []
    for query, gold in GOLD:
        gi = gold_index(gold)
        ranks = rank_fn(query) if kb_vecs is None else rank_fn(query, kb_vecs)
        if gi in ranks[:3]:
            r = ranks.index(gi) + 1
            recalls.append(1.0); mrrs.append(1.0 / r)
            detail.append(f"rank{r}")
        else:
            recalls.append(0.0); mrrs.append(0.0)
            detail.append(">3 ✗")
    return {"name": name, "recall": sum(recalls) / len(recalls),
            "mrr": sum(mrrs) / len(mrrs), "detail": detail}


if __name__ == "__main__":
    kb_vecs = embed_texts(KB)
    print("每条 query 的金标排名（1=第一名命中，>3=前3都没进）：\n")
    print(f"{'query':<28}{'向量':<10}{'关键词':<10}{'混合':<10}金标")
    results = [evaluate(vector_rank, "向量", kb_vecs),
               evaluate(keyword_rank, "关键词"),
               evaluate(rrf_rank, "混合", kb_vecs)]
    for (q, g), i in zip(GOLD, range(len(GOLD))):
        cells = [res["detail"][i] for res in results]
        print(f"{q:<30}{cells[0]:<12}{cells[1]:<12}{cells[2]:<12}{g}")

    print("\n汇总：")
    for res in results:
        print(f"  {res['name']}检索  Recall@3 = {res['recall']:.2f}   MRR = {res['mrr']:.2f}")
    print("\n自己看：哪几条 query 是关键词路救的？哪几条是向量路救的？"
          "\n混合路取长补短——这就是生产 RAG 的标配。")
