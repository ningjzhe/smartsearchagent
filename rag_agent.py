# rag_agent.py
import logging
import asyncio
from typing import List, Dict
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
import os
from dotenv import load_dotenv
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class RagAgent:
    def __init__(self, index_dir: str = "rag_index_store", model_config: Dict = None):
        """
        初始化RAG Agent
        :param index_dir: 索引存储目录
        :param model_config: 模型配置参数
        """
        self._setup_embedding_model()
        self._setup_llm()
        self.index_dir = index_dir
        self.model_config = model_config or {
            "temperature": 0.1,
            "max_tokens": 2000,
            "embed_batch_size": 10
        }
        self._initialize_index()

    def _initialize_index(self):
        """初始化向量索引"""
        try:
            storage_context = StorageContext.from_defaults(persist_dir=self.index_dir)
            self.index = load_index_from_storage(storage_context)
        except Exception as e:
            logger.error(f"索引加载失败: {str(e)}")
            raise RuntimeError(f"无法加载索引，请检查索引目录和文件完整性。错误: {str(e)}")

    def _setup_embedding_model(self):
        """配置Embedding模型"""
        Settings.embed_model = DashScopeEmbedding(
            model_name="text-embedding-v4",
            api_base=os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.getenv("ALI_API_KEY"),
            embed_batch_size=10
        )

    def _setup_llm(self):
        """配置LLM模型"""
        Settings.llm = OpenAILike(
            model="qwen3-max",
            api_base=os.getenv("DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=os.getenv("ALI_API_KEY"),
            is_chat_model=True,
            temperature=0.1,
            max_tokens=2000,
        )

    def retrieve_relevant_texts(self, question: str, top_k: int = 3) -> List[Dict]:
        """同步检索相关文本片段，使用索引的 retriever。"""
        try:
            logger.info(f"处理查询: {question}")

            retriever = self.index.as_retriever()
            docs = retriever.retrieve(question)

            if not docs:
                logger.warning("未找到相关网页")
                return [{"content": "未找到相关信息", "source": "RAG"}]

            formatted_results = []
            for doc in docs[:top_k]:
                # 兼容不同 Document 接口
                if hasattr(doc, "get_text"):
                    content = doc.get_text()
                else:
                    content = getattr(doc, "page_content", None) or str(doc)

                metadata = getattr(doc, "metadata", None) or getattr(doc, "extra_info", None) or getattr(doc, "node_info", None) or {}

                formatted_results.append({
                    "content": content,
                    "source": metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown",
                    "score": metadata.get("score", 0.0) if isinstance(metadata, dict) else 0.0,
                    "chunk_id": metadata.get("chunk_id", "") if isinstance(metadata, dict) else "",
                    "file_path": metadata.get("file_path", "") if isinstance(metadata, dict) else "",
                })

            logger.info(f"找到 {len(formatted_results)} 个相关片段")
            return formatted_results

        except Exception as e:
            logger.error(f"检索过程中发生错误: {str(e)}")
            return [{"content": "检索服务暂时不可用", "source": "RAG"}]

    # 使用 Settings.embed_model（已在 _setup_embedding_model 配置）来生成向量，
    # 因此无需在此处单独实现异步嵌入器。

    def rebuild_index(self, data_dir: str = None):
        """重建索引方法"""
        data_dir = data_dir or self.index_dir
        logger.info(f"开始重建索引，数据目录: {data_dir}")
        
        try:
            # 重新加载文档
            documents = SimpleDirectoryReader(data_dir, recursive=True).load_data()
            logger.info(f"加载了 {len(documents)} 个新网页")
            
            # 重建索引
            self.index = VectorStoreIndex.from_documents(documents)
            
            # 保存新索引
            self.index.storage_context.persist(persist_dir=self.index_dir)
            logger.info("索引重建完成")
        except Exception as e:
            logger.error(f"索引重建失败: {str(e)}")
            raise

# 示例使用
if __name__ == "__main__":
    # 初始化Agent
    agent = RagAgent()
    
    # 同步检索示例
    def demo():
        results = agent.retrieve_relevant_texts("我在2025年12月份做了什么事情")
        for result in results:
            print(f"来源: {result['source']}\n内容: {result['content'][:100]}...\n")

    demo()