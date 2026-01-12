import asyncio
import sys
from search_agent import AutonomousSearchAgent

class InteractiveChat:
    def __init__(self):
        self.agent = AutonomousSearchAgent()
        self.running = True
    
    def print_banner(self):
        print("=" * 60)
        print("           🤖 智能问答系统 (支持联网搜索和RAG)")
        print("=" * 60)
        print("命令说明：")
        print("  - 输入问题直接提问（由模型自主决定是否联网搜索和RAG）")
        print("  - 输入 '搜索回答:你的问题' 手动启用联网搜索")
        print("    例如：搜索回答：北京的天气如何？")
        print("  - 输入 '你的问题 RAG回答'手动启用RAG")
        print("    例如：我最近做了什么？ RAG回答")            
        print(" - 输入 '退出'、'quit' 或 'exit' 退出对话")
        print("-" * 60)
    
    async def process_command(self, user_input: str):
        user_input = user_input.strip()
        if user_input.lower() in ['quit', 'exit', '退出']:
            self.running = False
            return "再见！"
    
        # 检查是否同时包含"搜索回答："和" RAG回答"
        is_search = user_input.startswith('搜索回答：')
        is_rag = user_input.endswith('RAG回答')
        # 提取问题部分
        question = user_input
        if is_search:
            question = question[5:].strip()  # 移除"搜索回答："
        if is_rag:
            question = question[:-5].strip()  # 移除" RAG回答"
    
    
        # 如果问题为空，返回错误
        if not question:
            return "请输入搜索或RAG内容。"
    
        # 确定是否启用搜索和RAG
        response = await self.agent.ask(question, 
                                    enable_search=is_search, 
                                    enable_rag=is_rag,
                                    stream_callback=self.stream_callback)
        print()
        return response

    def stream_callback(self, content_chunk):
        """流式输出的回调函数，逐字打印内容"""
        print(content_chunk, end='', flush=True)

    async def run(self):
        """运行交互式聊天界面"""
        self.print_banner()
        
        while self.running:
            try:
                user_input = input("\n💬 请输入你的问题或命令：")
                if not user_input:
                    continue
                
                response = await self.process_command(user_input)
            except KeyboardInterrupt:
                print("\n\n接收到中断信号，正在退出...")
                self.running = False
            except Exception as e:
                print(f"\n❌ 发生错误：{str(e)}")


if __name__ == "__main__":
    chat = InteractiveChat()
    asyncio.run(chat.run())