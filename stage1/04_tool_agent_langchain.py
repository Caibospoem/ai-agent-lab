"""
第 3 课（下）：同一个 Agent，用 LangChain 1.x 写 —— 看清框架替你做了什么
================================================================
上一支脚本（03）你手写了 Agent 的全部器官；这一支用框架重写一遍。
对照着读，你就知道 LangChain 到底"封装"了什么（面试必考）：

    你在 03 手写的                    LangChain 1.x 对应物
    ─────────────────────────────    ─────────────────────────────────
    TOOL_SCHEMAS（手写JSON说明书） →  @tool 装饰器（读函数签名+docstring自动生成）
    OpenAI 客户端+兼容端点          →  ChatOpenAI（模型抽象层）
    run_agent() 的 while 循环       →  create_agent() 内部的 LangGraph 状态机
    messages 列表                   →  result["messages"]（含完整轨迹，思路相同）
    execute_tool() 分发器           →  框架按 tool_call 的 name 自动分发

所以框架的价值 = 省掉循环和分发的样板代码 + 提供记忆/中间件等扩展点。
但循环里每一步发生了什么，你已经在 03 亲手写过了——这就是先裸后框架的意义。

运行：python 04_tool_agent_langchain.py
无 key 时用 FakeListChatModel 验证图结构（不产生真实调用）。
"""
import json
import os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash")

# ---- 工具定义：@tool 自动生成 schema（docstring 就是模型的"说明书"）------
# 对比 03：那份手写 TOOL_SCHEMAS JSON 没了，函数签名+docstring 即说明书。
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前实时天气（气温/体感/天况/湿度）。用户问天气必须调用。"""
    try:
        r = httpx.get(
            f"https://wttr.in/{city}", params={"format": "j1"},
            headers={"User-Agent": "curl/8.0"}, timeout=10,
        )
        r.raise_for_status()
        cur = r.json()["current_condition"][0]
        return json.dumps({
            "city": city, "temp_c": cur["temp_C"], "desc": cur["weatherDesc"][0]["value"],
            "source": "wttr.in", "query_time": datetime.now().strftime("%H:%M"),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"天气查询失败: {e}"}, ensure_ascii=False)


@tool
def calculator(expression: str) -> str:
    """精确计算数学表达式，支持 + - * / % ** 和括号。涉及数字计算时必须调用。"""
    # 生产建议：复用 03 的 AST 白名单实现，这里从简演示框架结构
    allowed = set("0123456789+-*/%(). ")
    if not set(expression) <= allowed:
        return json.dumps({"error": "表达式含非法字符"}, ensure_ascii=False)
    try:
        return json.dumps({"expression": expression, "result": eval(expression)})  # 仅课堂演示
    except Exception as e:
        return json.dumps({"error": f"计算失败: {e}"}, ensure_ascii=False)


# ---- 组装 Agent ------------------------------------------------------
if API_KEY:
    # base_url 指向智谱 → 又一次印证：OpenAI 兼容格式是行业通用语言
    llm = ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=0.2)
else:
    print(">> Mock 模式：用 FakeListChatModel 验证图结构（无真实调用）\n")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    class MockLLM(FakeListChatModel):
        """细节：create_agent 的 model 节点会先 llm.bind_tools(工具) 完成挂载——
        这就是框架把工具说明书塞给模型的时机。Fake 模型没实现该接口，补一个。"""
        def bind_tools(self, tools, **kwargs):
            return self

    llm = MockLLM(responses=["[Mock回复] 图结构验证：模型节点正常返回。"])

agent = create_agent(
    llm,
    tools=[get_weather, calculator],
    system_prompt="你是助手'小智'：实时信息和精确计算必须用工具，回答注明来源，保持两句话以内。",
)

if __name__ == "__main__":
    questions = ["现在晋江天气怎么样？", "帮我精确计算 (3.7+1.2)*15"]

    for q in questions:
        print(f"\n你: {q}")
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        # result["messages"] = 完整轨迹，和你在 03 里手写的 messages 同一思想
        print("  ── 完整消息轨迹 ──")
        for m in result["messages"]:
            tcs = getattr(m, "tool_calls", None)
            extra = f"  → 调用 {[(tc['name'], tc['args']) for tc in tcs]}" if tcs else ""
            content = (m.content or "").strip()
            print(f"  [{type(m).__name__}] {content[:80]}{extra}")
