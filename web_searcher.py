import os
import sys
from dotenv import load_dotenv
import httpx
from fastmcp import FastMCP
import asyncio

# 加载环境变量
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("错误：未在环境变量中找到 TAVILY_API_KEY。请确认 .env 文件已配置。")

# 创建服务器实例
server = FastMCP("web_search")

# 定义工具函数
async def web_search(query: str, max_results: int = 5) -> str:
    """使用Tavily进行网络搜索，返回搜索结果的摘要"""
    api_url = "https://api.tavily.com/search"
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced"
    }
    # 修正点：将日志输出到 stderr，避免污染 MCP 协议通信的 stdout
    print(f" 服务器端：收到搜索请求 - '{query}'", flush=True, file=sys.stderr)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(api_url, json=payload)
            response.raise_for_status()
            search_data = response.json()

        print(" 服务器端：成功从Tavily获取数据", flush=True, file=sys.stderr)

        entries = search_data.get("results", [])[:max_results]
        if not entries:
            return "未找到相关结果。"

        summary_parts = []
        for i, entry in enumerate(entries, 1):
            summary_parts.append(
                f"【结果 {i}】\n"
                f"标题：{entry.get('title', 'N/A')}\n"
                f"内容：{entry.get('content', 'N/A')}\n"
                f"链接：{entry.get('url', 'N/A')}\n"
            )
        summary = "\n".join(summary_parts)
        return summary

    except httpx.HTTPError as e:
        error_msg = f"网络请求失败: {str(e)}"
        print(f" 服务器端：{error_msg}", flush=True, file=sys.stderr)
        return f"抱歉，搜索执行失败。原因：{error_msg}"
    except Exception as e:
        error_msg = f"处理搜索时发生未知错误: {str(e)}"
        print(f" 服务器端：{error_msg}", flush=True, file=sys.stderr)
        return f"抱歉，搜索执行失败。原因：{error_msg}"

# 修正点：使用正确的装饰器注册工具
@server.tool()
async def web_search_tool(query: str, max_results: int = 5) -> str:
    """使用Tavily进行网络搜索"""
    # 修正点：工具函数直接返回字符串，FastMCP 会自动将其包装成标准 MCP 响应
    result = await web_search(query, max_results)
    return result

# 修正点：简化服务器启动逻辑，使用内置的 run() 方法
if __name__ == "__main__":
    # 使用默认的 stdio 传输模式启动服务器
    server.run()