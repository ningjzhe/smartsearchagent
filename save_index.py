# save_index.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex ,Settings,StorageContext
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
INDEX_DIR = "rag_index_store"


ALI_API_KEY = os.environ.get("ALI_API_KEY")
Base_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

Settings.llm = OpenAILike(
    model="qwen3-max", 
    api_base=Base_URL,
    api_key=ALI_API_KEY,
    is_chat_model=True,
    # 以下参数可根据需要调整
    temperature=0.1,      # 控制创造性（0-1，值越小答案越确定）
    max_tokens=2000,     # 生成答案的最大长度
)
Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v4",
    api_base=Base_URL,
    api_key=ALI_API_KEY,
    embed_batch_size=10
)

def build_index():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"请把要索引的文档放到 {DATA_DIR} 目录后重新运行。")
        return

    # 1) 读取文档
    documents = SimpleDirectoryReader(DATA_DIR, recursive=True).load_data()
    print(f"成功加载了 {len(documents)} 个文档")

    # 2) 构建索引
    # 由于已经设置了 Settings.llm 和 Settings.embed_model，
    # VectorStoreIndex 会自动使用它们来生成文本向量和后续的查询
    index = VectorStoreIndex.from_documents(documents)  # 注意这里使用 VectorStoreIndex

    # 3) 保存索引到磁盘
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
    index.storage_context.persist(persist_dir=INDEX_DIR)
    print("索引已使用阿里云模型构建并保存到", INDEX_DIR)

if __name__ == "__main__":
    build_index()