"""
构建向量索引模块
使用 sentence-transformers 生成嵌入，存入 FAISS 向量数据库，并持久化到磁盘
"""
import os
import sys
# 将项目根目录添加到 sys.path，以便可以导入 src 模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 导入上一步的文档加载与切分函数
from src.load_docs import load_documents_from_folder, split_documents


def get_embedding_model():
    """
    返回一个 HuggingFaceEmbeddings 实例，使用轻量级多语言模型（支持中文）
    """
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings


def build_vector_store(docs: List[Document], embeddings, persist_directory: str):
    """
    从文档列表创建 FAISS 向量库，并持久化到本地文件夹
    """
    print(f"正在为 {len(docs)} 个文档块生成向量并创建索引...")
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(persist_directory)
    print(f"向量索引已保存到: {persist_directory}")
    return vector_store


def load_vector_store(persist_directory: str, embeddings):
    """
    从本地文件夹加载已有的 FAISS 索引
    """
    print(f"从 {persist_directory} 加载向量索引...")
    vector_store = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True)
    print("加载成功")
    return vector_store


if __name__ == "__main__":
    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "data")
    persist_dir = os.path.join(base_dir, "vector_store")

    # 1. 加载并切分文档
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"data 文件夹不存在: {data_folder}")
    raw_docs = load_documents_from_folder(data_folder)
    if not raw_docs:
        raise ValueError("data 文件夹中没有找到可支持的文档")
    chunked_docs = split_documents(raw_docs)

    # 2. 初始化嵌入模型
    print("\n正在加载嵌入模型（首次运行会下载模型，约 500MB，请耐心等待）...")
    embeddings = get_embedding_model()

    # 3. 构建向量库并保存
    vector_store = build_vector_store(chunked_docs, embeddings, persist_dir)

    # 4. 简单测试检索
    test_query = "什么是人工智能？"
    print(f"\n测试查询: {test_query}")
    results = vector_store.similarity_search(test_query, k=2)
    print(f"找到 {len(results)} 个相关文档块:")
    for i, doc in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        print(doc.page_content[:300])
        print(f"来源: {doc.metadata.get('source', '未知')}")