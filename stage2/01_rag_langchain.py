"""
第 4 课（阶段二开篇）：用 LangChain 标准件搭 RAG
================================================================
你在合规平台手写过完整 RAG（自算余弦相似度、阈值 0.33 软过滤、防幻觉引用校验），
所以这课不教"RAG 是什么"，而是对照框架：标准件和你手写的零件一一对应。

    你在合规平台手写的                 LangChain 标准件
    ────────────────────────────     ─────────────────────────────────────
    embedding-3 调 API + 向量列表   →  OpenAIEmbeddings（Embeddings 接口）
    手写余弦相似度 + top-k 排序     →  InMemoryVectorStore.as_retriever()
    拼字符串塞 prompt               →  ChatPromptTemplate({context} 占位)
    "只能引用召回内容"约束          →  写进 prompt 的"期望边界"（思想完全一致）

为什么需要 RAG？（回收第2课思考题3）上下文窗口放不下所有知识 + 按需检索省 token
+ 答案可溯源。一句话：**上下文窗口小，才需要 RAG。**

运行：python 01_rag_langchain.py    （无 key 自动 Mock，链路结构照跑）
     python 01_rag_langchain.py chat 进入交互问答
"""
import os
import sys

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
LLM_MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash")
EMBED_MODEL = "embedding-3"  # 你 MT30 项目同款，语义检索已验证可用

# ---- 第 1 部分：知识库（生产中来自文档库；这里 8 条演示）----------------
KB_DOCS = [
    Document(page_content="本训练营课程核心公式：Agent = LLM + 工具 + 循环。"),
    Document(page_content="训练营收取学员的门槛是：完成上一课作业并通过思考题考核。"),
    Document(page_content="训练营阶段二主题：RAG 与记忆策略，共 3 课（第4-6课）。"),
    Document(page_content="训练营阶段三主题：LangGraph 状态机与多 Agent 协作，共 4 课。"),
    Document(page_content="GLM-4-Flash 是智谱的免费对话模型，支持 Function Calling。"),
    Document(page_content="智谱 embedding-3 是文本向量化模型，用于语义检索。"),
    Document(page_content="wttr.in 是免费无 key 的天气 API，第 3 课用作 Agent 工具。"),
    Document(page_content="MCP 是 Anthropic 提出的工具接入协议，比喻为 Agent 的 USB-C。"),
]

# ---- 第 2 部分：向量库与检索器 -------------------------------------------
def build_components():
    """有 key 用真模型，无 key 用 Fake 验证链路（接口一致——接口抽象的价值）"""
    if API_KEY:
        # check_embedding_ctx_length=False：绕过 tiktoken 预分词，把原文直接发给智谱
        embeddings = OpenAIEmbeddings(
            model=EMBED_MODEL, base_url=BASE_URL, api_key=API_KEY,
            check_embedding_ctx_length=False,
        )
        llm = ChatOpenAI(model=LLM_MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.2)
    else:
        print(">> Mock 模式：Fake 向量 + Fake 模型，仅验证链路\n")
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        embeddings = DeterministicFakeEmbedding(size=256)
        llm = FakeListChatModel(responses=["[Mock回复] 链路验证正常。"])

    # from_documents = 内部逐条 embed + 存入内存库（你在合规平台手写的入库循环）
    vectorstore = InMemoryVectorStore.from_documents(KB_DOCS, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 标准件版"余弦top3"
    return vectorstore, retriever, llm


vectorstore, retriever, llm = build_components()

# ---- 第 3 部分：RAG prompt（期望边界 = 你合规平台"引用校验"的 prompt 版）--
SYSTEM_TMPL = (
    "你是知识库问答助手。回答规则：\n"
    "1. 只根据下面的参考片段回答，并在答案末尾标注引用了哪个片段；\n"
    "2. 如果参考片段不足以回答，直接回复\"知识库中没有相关内容\"，禁止编造。\n\n"
    "参考片段：\n{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_TMPL), ("human", "{question}")]
)


def format_docs(docs) -> str:
    return "\n\n".join(f"[片段{i+1}] {d.page_content}" for i, d in enumerate(docs))


# ---- 第 4 部分：LCEL 链 —— 数据像管道一样流过每个组件 -------------------
# {"context": 检索结果, "question": 原问题} → prompt → llm → 文本
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---- 第 5 部分：演示入口 -------------------------------------------------
def answer(question: str) -> str:
    # 先把检索过程打印出来 —— 检索质量决定 RAG 上限，要学会盯这一步
    hits = vectorstore.similarity_search_with_score(question, k=3)
    print("  [检索] 命中片段（相似度）:")
    for doc, score in hits:
        print(f"    {score:.3f}  {doc.page_content[:36]}…")
    return rag_chain.invoke(question)


if __name__ == "__main__":
    demo_questions = [
        "训练营阶段二学什么？",          # 库内有答案 → 正常回答并引用
        "老板的银行卡密码是多少？",      # 库外问题 → 必须拒答（期望边界）
    ]
    for q in demo_questions:
        print(f"\n问: {q}")
        print(f"答: {answer(q)}")

    # 交互模式：python 01_rag_langchain.py chat
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        print("\nRAG 问答已就绪 | q 退出")
        while True:
            try:
                q = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("q", "quit", "exit") or not q:
                break
            print(f"答: {answer(q)}\n")
