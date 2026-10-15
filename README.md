# AI Agent 实战训练营（求职导向修正版）

> 原计划：三阶段 7 个项目。本 README 是修正后的执行版，边学边更新进度。

## 一、为什么修正原计划（求职视角）

| # | 原计划 | 修正 | 原因 |
|---|--------|------|------|
| 1 | 直接上 LangChain 教程 | **先裸 API → 再框架** | 面试必问"LangChain 帮你做了什么"，不懂裸 API 的人答不出 |
| 2 | 用旧教程（AgentExecutor / Zero-Shot ReAct） | **LangChain 1.x + LangGraph** | 1.0 重构后 AgentExecutor 已移除；市场主流是 LangGraph。旧概念（ReAct 模式）仍要懂，作为原理学习 |
| 3 | RAG 从零学 | **对比学习**：你已有手写 RAG（合规平台） | 用 LangChain 标准件重写一遍，对比差异 → 直接变成面试亮点故事 |
| 4 | Tavily/SerpAPI 搜索 | **DuckDuckGo（免费）起步** | Tavily 要注册收费，学习阶段没必要 |
| 5 | CrewAI / LangGraph 二选一 | **主攻 LangGraph，CrewAI 了解即可** | LangGraph 是 2025-2026 招聘 JD 高频词；CrewAI 岗位占比小 |
| 6 | （缺失） | **补充：Function Calling 底层原理、Agent 评测、MCP 协议** | 面试高频考点，教程通常不讲 |

## 二、执行路线（4 周，每天 60-90 分钟）

```
W1 阶段一：本质理解
  ├─ 第1课 裸 API 调用 LLM            ← 你在这里
  ├─ 第2课 多轮对话循环 = 记忆雏形      ← 你在这里
  └─ 第3课 工具调用 Agent（Function Calling + ReAct 原理）
W2 阶段二：标准件
  ├─ 第4课 RAG Agent（LangChain 版，对比你的手写版）
  ├─ 第5课 会话记忆（Buffer / Summary / 截断策略）
  └─ 第6课 Web 研究 Agent（搜索工具 + 综合回答）
W3 阶段三：状态机
  ├─ 第7课 LangGraph 入门（图、节点、边、状态）
  └─ 第8课 用 LangGraph 重写研究 Agent（计划→搜索→综合）
W4 冲刺：多 Agent + 求职包装
  ├─ 第9课 多 Agent 协作（角色分工 + 消息传递）
  ├─ 第10课 综合项目：领域专家 Agent（建议结合合规/代码课题）
  └─ 面试包装：GitHub 整理 + 简历话术 + 高频问题演练
```

## 三、进度追踪

- [x] 第1课 裸 API 调用（已本地复现 ✅ 2026-10-11）
- [x] 第2课 多轮对话循环（已本地复现 ✅ 作业B通过；思考题过关后发第3课）
- [x] 第3课 工具调用 Agent（已本地复现 ✅ 2026-10-13：now_time工具+多工具链）
- [x] 第4课 RAG（已交付 2026-10-15：00手写版+01标准件版两级对照，待本地复现）
- [ ] 第7-8课 LangGraph
- [ ] 第9-10课 多 Agent + 综合项目

## 四、你的本地环境配置（Windows + conda）

```bash
# 1. 用你现有的 conda 环境（或新建一个更干净）
conda activate chain
# 2. 装依赖
pip install -r requirements.txt
# 3. 配置 key（免费）
#    去 https://open.bigmodel.cn 注册 → 控制台 → API keys → 创建并复制
#    把 ai-agent-lab/.env.example 复制为 .env，填入你的 key
# 4. 验证
python stage1/01_hello_llm.py
python stage1/02_chat_loop.py
```

## 五、学习约定（手把手模式的规则）

每课固定节奏，缺一步不准进入下一课：

1. **跑通**：本地运行课程代码，观察输出
2. **改一次**：按课后作业要求修改代码（改过才真是你的）
3. **答思考题**：回复 AI 助手你的答案，答对才发下一课

> 工程原则：先跑通再理解，先理解再扩展。代码必须自己敲一遍关键部分。
