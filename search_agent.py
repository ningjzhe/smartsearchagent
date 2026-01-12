import os
import sys
import time
import asyncio
import traceback
from typing import Dict, List, Optional
from openai import OpenAI
import json
import subprocess
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import StdioServerParameters
from enum import Enum
from typing import Dict, Any, Optional,List
import traceback
from rag_agent import RagAgent  # 相对导入（需确保包结构）
load_dotenv()
script_path = os.path.join(os.path.dirname(__file__), "web_searcher.py")
class AutonomousSearchAgent:
    
    def __init__(self, model_name: str = "qwen-max", index_dir: str = "rag_index_store"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("ALI_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.conversation_history = []
        self.rag_agent = RagAgent(index_dir)
    async def ask(self, question: str, enable_search: bool = False, enable_rag: bool = False, stream_callback=None) -> str:
        """处理用户问题的主入口
        enable_search: True=强制搜索,False=由AI自动判断
        enable_rag: True=强制RAG, False=由AI自动判断
        """
        print(f"\n🤔 用户问题：{question}")
        
        # 确定是否启用搜索和RAG
        final_search_decision = enable_search
        final_rag_decision = enable_rag
        
        # 如果未指定，由AI自动判断
        if not final_search_decision:
            ai_decision = await self._need_search_by_ai(question)
            final_search_decision = ai_decision
            print(f"🌐 搜索决策：AI自动判断 -> {'启用' if final_search_decision else '禁用'}")
        
        # 如果未指定，由AI自动判断RAG
        if not final_rag_decision:
            ai_rag_decision = await self._need_rag_by_ai(question)
            final_rag_decision = ai_rag_decision
            print(f"📚 RAG决策：AI自动判断 -> {'启用' if final_rag_decision else '禁用'}")
        
        # 获取所有可用的上下文
        context = await self._get_context(question, final_search_decision, final_rag_decision)
        
        # 使用整合后的上下文回答问题
        return await self._answer_with_context(question, context, stream_callback)
    
    async def _get_context(self, question: str, enable_search: bool, enable_rag: bool) -> str:
        """获取所有可用的上下文（搜索和RAG）"""
        context_parts = []
        
        # 获取搜索结果
        if enable_search:
            print("🔍 正在执行搜索...")
            search_result = await self.real_web_search(question)
            if search_result and search_result != "搜索工具未返回有效结果":
                context_parts.append(f"### 搜索结果\n{search_result}")
        
        # 获取RAG结果
        if enable_rag:
            print("📚 正在执行RAG检索...")
            rag_result = await self._retrieve_rag(question)
            if rag_result and rag_result != "RAG检索未返回有效结果":
                context_parts.append(f"### RAG检索结果\n{rag_result}")
        
        # 返回整合后的上下文
        if context_parts:
            return "\n\n".join(context_parts)
        else:
            return "没有可用的额外信息。"
    
    async def _answer_with_context(self, question: str, context: str, stream_callback=None) -> str:
        """使用整合后的上下文回答问题"""
        try:
            # 构建对话历史
            system_prompt = f"你是一个有帮助的AI助手，可以结合以下上下文回答问题。请严格基于提供的上下文信息回答问题，并注明信息来源。上下文：\n{context}"
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": question})
            
            full_response = ""
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                if stream_callback:
                    stream_callback(content)
            
            # 更新对话历史
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
            # 保持对话历史长度
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            return full_response
        except Exception as e:
            return f"回答问题时出现错误：{str(e)}"
    
    async def _need_search_by_ai(self, question: str) -> bool:
        """通过AI模型判断问题是否需要联网搜索"""
        system_prompt = """
你是一个搜索决策助手。你的任务仅是判断用户的问题是否还需要联网搜索以获取最新信息。
请仅根据问题内容本身和当前已经获得的搜索结果进行判断，无需回答问题。
# 判断规则
## 必须搜索的情况（返回 yes）：
- 为回答当前问题，需要进一步搜索
## 不用继续搜索的情况，（返回 no）：
- 问题可以直接回答，或者已经有足够的信息回答
- 问题涉及的是私人知识，网络搜索无法得到答案
你的回答必须严格遵循以上规则。请只回答 'yes' 或 'no'。
"""
        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": question})
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=10,
                temperature=0.0
            )
            decision = response.choices[0].message.content.strip().lower()
            return 'yes' in decision
        except Exception as e:
            print(f"意图识别失败，默认不搜索。错误: {e}")
            return False
    
    async def _need_rag_by_ai(self, question: str) -> bool:
        """通过AI模型判断问题是否需要RAG"""
        system_prompt = """
你是一个RAG决策助手。你的任务仅是判断用户的问题是否需要RAG（检索增强生成）以获取更准确的答案。
请仅根据问题内容本身进行判断，无需回答问题。
现在的RAG知识库是关于提问者自己的个人知识的。
# 判断规则
## 必须使用RAG的情况（返回 yes）： 
- 问题涉及提问者的个人经历、活动、计划等专有信息。
- 需要结合提问者的个人背景知识来回答的问题。
## 不需使用RAG的情况（返回 no）：
- 通用知识、公共信息、概念解释、数学计算、代码编写。
- 文本处理任务，如翻译、摘要、润色。
你的回答必须严格遵循以上规则。
请只回答 'yes' 或 'no'。
"""
        try:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": question})
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=10,
                temperature=0.0
            )
            decision = response.choices[0].message.content.strip().lower()
            return 'yes' in decision
        except Exception as e:
            print(f"RAG意图识别失败，默认不使用RAG。错误: {e}")
            return False
    async def _retrieve_rag(self, question: str) -> str:
        """从索引库提取相关文本片段"""
        return self.rag_agent.retrieve_relevant_texts(question)
    
    async def real_web_search(self, query: str, max_results: int = 5, max_search_attempts: int = 3) -> str:
        """
        使用MCP框架直接调用web_search工具进行真正的网络搜索,并进行多轮搜索优化查询。
        """
        search_attempts=0
        search_continue=True
        search_results=[]
        question=query
        while search_attempts < max_search_attempts and search_continue:
            try:
                print("🔄 准备进行下一轮搜索。")
                search_attempts += 1
                question=await self._furthur_ask(query,search_brief=search_results) 
                print(f"🔄 下一轮搜索问题：{question}")
                # 优化查询
                optimized_query = await self._optimize_search_query(query,search_brief=search_results,query=question)
            
                # 配置MCP服务器参数
                server_params = StdioServerParameters(
                    command=sys.executable,
                    args=[script_path]  # 请替换为您的MCP服务器脚本路径
                )
                print(f"🔗 正在连接MCP服务器执行搜索: {optimized_query}")
            
                # 使用stdio客户端连接MCP服务器
                async with stdio_client(server_params) as (read, write):
                    await asyncio.sleep(0.5)  # 等待服务器完全启动
                    async with ClientSession(read, write) as session:
                        print("✅ ClientSession创建成功")
                        await session.initialize()
                        print("✅ 会话初始化完成")
                    
                        # 执行搜索
                        result = await session.call_tool(
                            "web_search_tool",
                            {
                                "query": optimized_query,
                                "max_results": max_results
                            }
                        )
                        # 提取工具返回的内容
                        if result.content:
                            search_result = "\n".join(
                                content.text for content in result.content 
                                if hasattr(content, 'text')
                            )
                        else:
                            return "搜索工具未返回有效结果"     
                        search_brief= await self._analyze_search_result(search_result, query,search_results,question)
                        search_results.append(search_brief)
                        print(f"📝 本轮搜索结果摘要：{search_brief}")
                        # 判断是否需要继续搜索
                        search_continue = await self._need_search_by_ai(f"基于当前搜索结果摘要：{search_results}，请判断是否需要继续搜索以获取更多信息来回答用户的原始查询：{query}？请只回答 'yes' 或 'no'。")
            
            except Exception as e:
                error_msg = f"MCP搜索连接失败: {str(e)}"
                traceback.print_exc()
                print(f"❌ {error_msg}")
                return error_msg
        print("✅ 搜索完成")
        return "\n".join(search_results)
    async def _furthur_ask(self, original_query: str, search_brief: list = None) -> str:
        """生成搜索agent进一步要查询的信息"""
        try:
            system_prompt = """
你是一个搜索问题生成助手。你的任务是根据当前的搜索结果和用户原始问题，生成下一步需要进一步查询的问题。
你的回答必须严格遵循以上规则。请只返回生成的问题，不要包含其他多余信息。
"""
            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "user", "content": f"用户原始查询：{original_query}\n之前的搜索结果的摘要：{search_brief}\n请生成下一步需要进一步查询的问题。"})
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成进一步查询问题失败。错误: {e}")
            return original_query
    async def _analyze_search_result(self, search_result: str, original_query: str,seaech_brief,question) -> str:
        """总结搜索结果"""
        try:
            system_prompt = """
你是一个搜索结果总结助手。你的任务是针对当前查询的问题，总结当前的搜索结果，写出搜索结果的摘要。
你的回答必须严格遵循以上规则。请只返回总结内容，不要包含其他多余信息。
"""
            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "user", "content": f"用户原始查询：{original_query}\n之前的搜索结果的摘要：{seaech_brief}\n,当前查询的问题：{question}\n新得到的搜索结果：{search_result}，\n请针对当前查询的问题和用户的原始提问，总结新得到的搜索结果，返回总结内容"})
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3
            )
            search_brief = response.choices[0].message.content.strip().lower()
            return search_brief
        except Exception as e:
            print(f"搜索结果总结失败。错误: {e}")
            return "总结搜索结果失败"
    async def _optimize_search_query(self, original_query: str, strategy: str = "semantic_rewrite", search_brief: list = None, query: str = None) -> str:
        """使用大模型优化搜索查询"""
        try:
            if strategy == "semantic_rewrite":
                system_prompt = """
你是一个专业的搜索查询优化助手。你将获得的信息为用户的原始问题，目前得到的搜索摘要，以及搜索agent进一步要查询的信息\n你的任务是根据已有的信息将搜索agent的问题（如果没有，就对用户的问题）重写为明确、具体、适合搜索引擎检索的查询。
注意：你需要返回的是针对问题的适合搜索引擎的查询，而不是回答问题。
# 优化规则
- 保持原意：不能改变问题的核心意图，不得根据模型已有的过时信息修改用户问题。
- 具体化：将模糊表述转为明确关键词（如"最近"→"2026年"）
- 结构化：包含关键实体、时间、地点等限定信息
- 搜索友好：使用搜索引擎容易匹配格式
- 避免冗余：去除无关词汇，保持简洁
请根据以上规则优化查询。 
"""
            else:
                return original_query  # 未知策略则返回原问题
            
            messages = [{"role": "system", "content": system_prompt}]
            content=f"用户原始查询：{original_query}\n已经得到的搜索结果简要：{search_brief}\n搜索agent需要进一步查询的信息:{query if query else original_query},请优化为适合搜索引擎的查询。"
            messages.append({"role": "user", "content": content})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=50,
                temperature=0.1
            )
            optimized_query = response.choices[0].message.content.strip()
            return optimized_query
        except Exception as e:
            print(f"查询优化失败，使用原始查询: {str(e)}")
            return original_query