"""
第 3 课（上）：工具调用 Agent —— 裸 SDK 版
================================================================
本课解决你作业里的真实痛点：glm-4-flash 编造晋江天气（先说查不了，
换个问法又编出"18-25℃"）。原因：模型只有文本预测能力，没有事实接口。
解法：给它工具，让它自己决定"先查再答"。

【面试标准答案】Agent = LLM + 工具 + 循环
  LLM 负责决策（调哪个工具、传什么参数），但不执行任何工具；
  执行发生在你的客户端代码里；执行结果回填对话历史，模型读后继续。

一次工具调用的完整时序（对照下面的代码看）：
    你: "晋江天气如何？"
        │ ① messages 发给模型
        ▼
    模型返回: tool_calls=[get_weather({"city":"晋江"})]   ← 只是"意图"，JSON而已
        │ ② 你执行真正的 HTTP 请求（wttr.in）
        ▼
    工具结果 → 以 role=tool 追加进 messages                ← ③ 回填，tool_call_id 必须对上
        │ ④ 带着结果再问模型
        ▼
    模型: "晋江现在晴，25℃（来源：wttr.in）"              ← 终态，循环结束

【新增的两种消息角色】（第2课讲过 system/user/assistant）
  assistant + tool_calls：模型的"我想调用工具"意图
  tool                  ：工具执行结果，必须紧跟对应的 assistant 消息

运行：python 03_tool_call_bare.py   （无 key 自动进 Mock 模式，结构照跑）
命令：q 退出 | reset 清空 | history 打印完整消息列表
"""
import ast
import json
import operator
import os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY", "").strip()
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash")

SYSTEM_PROMPT = (
    "你是助手'小智'。你拥有工具：需要实时信息（天气）或精确计算时必须先调用工具，"
    "禁止凭记忆编造数据；拿到工具结果后用一两句话作答并注明数据来源。"
)

# ---- 第 1 部分：工具的"说明书"（发给模型看的 schema）--------------------
# 模型读的是这段 JSON 描述来决定何时用工具、参数怎么填——description 写得越
# 清楚，模型选得越准。这也是为什么说"写工具 = 写好函数 + 写好说明书"。
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前实时天气（气温/体感/天况/湿度）",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如：晋江、上海、Tokyo"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "精确计算数学表达式，支持 + - * / % ** 和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "合法算式，如 (3.7+1.2)*15"}
                },
                "required": ["expression"],
            },
        },
    },
]

# ---- 第 2 部分：工具的真实实现（跑在你本机，不在模型那边）---------------
def get_weather(city: str) -> str:
    """wttr.in：免费、无需 key 的天气 API。工具结果统一返回 JSON 字符串。"""
    try:
        r = httpx.get(
            f"https://wttr.in/{city}",
            params={"format": "j1"},
            headers={"User-Agent": "curl/8.0"},
            timeout=10,
        )
        r.raise_for_status()
        cur = r.json()["current_condition"][0]
        return json.dumps(
            {
                "city": city,
                "temp_c": cur["temp_C"],
                "feels_like_c": cur["FeelsLikeC"],
                "desc": cur["weatherDesc"][0]["value"],
                "humidity": cur["humidity"] + "%",
                "source": "wttr.in",
                "query_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
            ensure_ascii=False,
        )
    except Exception as e:  # 失败也返回 JSON —— 错误要回喂给模型让它自己决策
        return json.dumps({"error": f"天气查询失败: {e}"}, ensure_ascii=False)


# calculator 用 AST 白名单解析，绝不裸用 eval()（面试安全考点：eval 任意代码执行）
_CALC_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _CALC_OPS:
        return _CALC_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"表达式含不允许的元素: {ast.dump(node)[:60]}")


def calculator(expression: str) -> str:
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
        return json.dumps({"expression": expression, "result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"计算失败: {e}"}, ensure_ascii=False)


def execute_tool(name: str, arguments_json: str) -> str:
    """工具分发器。arguments 是 JSON 字符串（经典坑：不是字典，要手动 loads）"""
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "参数不是合法JSON"}, ensure_ascii=False)
    if name == "get_weather":
        return get_weather(args.get("city", ""))
    if name == "calculator":
        return calculator(args.get("expression", ""))
    return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)


# ---- 第 3 部分：与模型对话（真实/Mock 双模式，返回统一结构）-------------
def chat(messages: list[dict]) -> dict:
    """统一返回 {"content": str|None, "tool_calls": [{"id","name","arguments"}]}"""
    if API_KEY:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS, temperature=0.2,
        )
        msg = resp.choices[0].message
        tcs = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in (msg.tool_calls or [])
        ]
        return {"content": msg.content, "tool_calls": tcs}

    return _mock_chat(messages)


def _mock_chat(messages: list[dict]) -> dict:
    """Mock：模拟两次循环 —— 先吐工具调用意图，收到 tool 消息后吐最终回答"""
    last = messages[-1]
    if last["role"] == "tool":  # 第二轮：已有工具结果 → 给终态回答
        return {"content": f"[Mock回复] 已拿到工具结果 {last['content'][:50]}…，这就是最终回答。", "tool_calls": []}

    text = last["content"]
    if any(c in text for c in "+-*/") and any(ch.isdigit() for ch in text):
        import re
        m = re.search(r"[-+*/%().\d\s]{3,}", text)
        expr = (m.group(0).strip() if m else "1+1")
        return {"content": None, "tool_calls": [
            {"id": "mock_1", "name": "calculator", "arguments": json.dumps({"expression": expr})}
        ]}
    if "天气" in text or "气温" in text or "温度" in text:
        import re
        # 去掉指令动词后再提取城市名（Mock专用启发式；真实模型自己就会填对参数）
        stripped = re.sub(r"帮我|请|麻烦|我|查一下|查询|查|看看|看下|一下", "", text)
        m = re.search(r"([\u4e00-\u9fa5]{2,6}?)(?:的)?(?:当前|现在)?(?:天气|气温|温度)", stripped)
        city = m.group(1) if m else "晋江"
        return {"content": None, "tool_calls": [
            {"id": "mock_1", "name": "get_weather", "arguments": json.dumps({"city": city}, ensure_ascii=False)}
        ]}
    return {"content": "[Mock回复]（没有 key，我只会处理天气/计算类问题）", "tool_calls": []}


# ---- 第 4 部分：Agent 循环 —— 本课的主角，请逐行读懂 --------------------
MAX_STEPS = 6  # 防失控：真实系统必须有循环上限（面试工程化考点）


def run_agent(messages: list[dict]) -> str:
    for step in range(1, MAX_STEPS + 1):
        resp = chat(messages)

        if not resp["tool_calls"]:  # 没有工具调用 → 模型给出了最终回答，退出循环
            return resp["content"]

        for tc in resp["tool_calls"]:
            print(f"  [Agent·step{step}] 调用 {tc['name']}({tc['arguments']})")

            # ① 模型的调用"意图"必须追加进历史（和第2课追加回复同理：不追加=失忆）
            messages.append({
                "role": "assistant",
                "content": resp["content"],
                "tool_calls": [{
                    "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }],
            })
            # ② 在客户端真正执行工具
            result = execute_tool(tc["name"], tc["arguments"])
            print(f"  [工具结果] {result}")
            # ③ 结果以 role=tool 回填，tool_call_id 必须和调用意图对上
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

    return "（已达到最大工具调用步数，强制停止以保护系统）"


# ---- 第 5 部分：交互入口（和第2课同一副骨架，换了大脑）------------------
def main() -> None:
    if not API_KEY:
        print(">> Mock 模式：填入 key 后用真工具查真天气\n")
    print(f"Agent 已就绪（{MODEL}） | 工具: get_weather, calculator | q=退出 reset=清空 history=看历史\n")

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
            print(">> 记忆已清空\n")
            continue
        if user_input.lower() == "history":
            print(json.dumps(messages, ensure_ascii=False, indent=2), "\n")
            continue

        messages.append({"role": "user", "content": user_input})
        reply = run_agent(messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"小智: {reply}    [当前历史 {len(messages)} 条]\n")


if __name__ == "__main__":
    main()
