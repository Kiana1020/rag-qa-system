"""
问答链模块
加载向量库和 Ollama 模型，实现基于文档的智能问答
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def get_embedding_model():
    """与 build_index.py 中相同的嵌入模型，用于将问题转换成向量"""
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def load_vector_store(persist_directory: str, embeddings):
    """加载本地 FAISS 索引"""
    print(f"从 {persist_directory} 加载向量库...")
    vector_store = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store


def create_qa_chain(vector_store, model_name="qwen2:1.5b", temperature=0.1):
    """
    创建 RetrievalQA 链
    - vector_store: 向量数据库
    - model_name: Ollama 模型名称（需要先 pull）
    - temperature: 控制随机性，越低答案越确定（避免瞎编）
    """
    # 初始化 Ollama 模型
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        # 可以添加其他参数，比如 top_p、num_predict 等
    )

    # 自定义提示模板，强制模型基于提供的上下文回答
    prompt_template = """你是一个基于文档的问答助手。请仅根据以下【文档内容】来回答问题。
如果文档内容中没有答案，请直接回答"根据现有文档无法回答该问题"，不要编造信息。

【文档内容】
{context}

【问题】
{question}

【回答】"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # 创建检索问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 将所有相关文档块拼接后一次性发给模型
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),  # 检索3个最相关块
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,  # 方便调试查看来源
    )
    return qa_chain


def main():
    # 路径配置
    base_dir = os.path.dirname(os.path.dirname(__file__))
    persist_dir = os.path.join(base_dir, "vector_store")

    # 加载嵌入模型
    print("加载嵌入模型...")
    embeddings = get_embedding_model()

    # 加载向量库
    vector_store = load_vector_store(persist_dir, embeddings)

    # 创建问答链（确认你的 Ollama 中已下载 qwen2:1.5b，且服务正在运行）
    print("初始化 Ollama 模型（确保 Ollama 后台运行中）...")
    qa_chain = create_qa_chain(vector_store, model_name="qwen2:1.5b", temperature=0.1)

    print("\n智能文档问答系统已就绪！输入问题（输入 exit 退出）\n")
    while True:
        query = input("你: ")
        if query.lower() in ["exit", "quit", "退出"]:
            break
        if not query.strip():
            continue

        print("思考中...")
        try:
            result = qa_chain.invoke({"query": query})
            answer = result["result"]
            source_docs = result["source_documents"]

            print(f"\n助手: {answer}\n")
            print("--- 参考来源 ---")
            for i, doc in enumerate(source_docs):
                src = doc.metadata.get("source", "未知")
                print(f"{i + 1}. [{src}] {doc.page_content[:100]}...")
            print()
        except Exception as e:
            print(f"出错了: {e}")


if __name__ == "__main__":
    main()