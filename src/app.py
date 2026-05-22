"""
Streamlit Web 界面：智能文档问答助手
支持上传文档（.txt, .pdf, .docx），动态构建索引，并回答基于文档的问题
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 导入之前的文档加载和切分函数
from src.load_docs import load_documents_from_folder, split_documents

# 页面配置
st.set_page_config(page_title="智能文档问答助手", page_icon="📚", layout="wide")
st.title("📚 智能文档问答助手 (RAG)")
st.markdown("上传你的文档（.txt, .pdf, .docx），然后提出任何问题，答案将基于文档内容生成。")

# 侧边栏：全局设置
with st.sidebar:
    st.header("⚙️ 设置")
    model_name = st.selectbox(
        "选择 Ollama 模型",
        options=["qwen2:1.5b", "qwen2:0.5b", "llama3.2:3b"],
        index=0,
        help="需要先在本地通过 `ollama pull` 下载模型"
    )
    temperature = st.slider("回答创造性 (temperature)", 0.0, 1.0, 0.1, 0.05)
    top_k = st.slider("检索相关文档块数量 (k)", 1, 5, 3, 1)

    st.header("📁 当前使用的文档")
    st.info("首次使用请上传文档，系统会自动构建索引。上传后等待几秒钟。")

    # 上传文档区域（支持多文件）
    uploaded_files = st.file_uploader(
        "上传文档（支持 .txt, .pdf, .docx）",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )

# 初始化会话状态（保存 vector_store 和 embedding 模型，避免重复加载）
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedding_model" not in st.session_state:
    # 注意：嵌入模型必须与之前构建时使用的模型一致
    st.session_state.embedding_model = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "current_docs_hash" not in st.session_state:
    st.session_state.current_docs_hash = ""


def get_embedding_model():
    """返回嵌入模型（单例模式）"""
    if st.session_state.embedding_model is None:
        model_name_emb = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        st.session_state.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name_emb,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    return st.session_state.embedding_model


def build_vector_store_from_files(uploaded_files):
    """将上传的文件保存到临时目录，加载、切分、构建 FAISS 向量库"""
    # 创建临时文件夹
    temp_dir = tempfile.mkdtemp()
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # 加载文档
    docs = load_documents_from_folder(temp_dir)
    if not docs:
        st.error("没有加载到任何有效文档，请检查文件格式或内容。")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    # 切分
    chunks = split_documents(docs)
    # 生成向量库
    embeddings = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embeddings)
    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)
    return vector_store


def create_qa_chain(vector_store, model_name, temperature, top_k):
    """创建问答链"""
    llm = ChatOllama(model=model_name, temperature=temperature)
    prompt_template = """你是一个基于文档的问答助手。请仅根据以下【文档内容】来回答问题。
如果文档内容中没有答案，请直接回答"根据现有文档无法回答该问题"，不要编造信息。

【文档内容】
{context}

【问题】
{question}

【回答】"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": top_k}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return qa_chain


# 处理上传的文档
if uploaded_files:
    # 简单哈希，用于检测文档是否变化（此处用文件名列表+修改时间？简化处理：只要上传了非空文件就重建）
    with st.spinner("正在处理上传的文档，构建索引中（可能需要几分钟）..."):
        vector_store = build_vector_store_from_files(uploaded_files)
        if vector_store is not None:
            st.session_state.vector_store = vector_store
            # 更新问答链
            st.session_state.qa_chain = create_qa_chain(
                vector_store, model_name, temperature, top_k
            )
            st.success(f"成功处理 {len(uploaded_files)} 个文档，索引已就绪！")
        else:
            st.error("文档处理失败，请重试。")

# 如果已有向量库（可能是之前上传并构建的），检查是否需要更新模型参数（temperature 或 k 变化）
if st.session_state.vector_store is not None:
    # 当用户改变模型参数时，重新创建 qa_chain（不重建向量库）
    if st.session_state.qa_chain is None or st.button("刷新问答链（应用新设置）"):
        with st.spinner("更新问答链..."):
            st.session_state.qa_chain = create_qa_chain(
                st.session_state.vector_store,
                model_name,
                temperature,
                top_k
            )
            st.success("设置已应用")

# 主界面：问答区域
if st.session_state.qa_chain is None:
    st.info("👈 请先在侧边栏上传文档，系统会自动构建索引。")
else:
    st.header("💬 提问")
    query = st.text_input("输入你的问题:", placeholder="例如：文档中提到了什么技术？")
    if query:
        if st.button("提交问题", type="primary"):
            with st.spinner("正在检索并生成回答..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": query})
                    answer = result["result"]
                    source_docs = result["source_documents"]

                    st.markdown("### 🤖 回答")
                    st.write(answer)

                    with st.expander("📄 参考来源"):
                        for i, doc in enumerate(source_docs):
                            src = doc.metadata.get("source", "未知文件")
                            st.markdown(f"**来源 {i + 1}: {src}**")
                            st.text(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
                except Exception as e:
                    st.error(f"出错了: {e}")

# 页脚
st.markdown("---")
st.caption("基于 LangChain + FAISS + Ollama | 模型运行在本地，保护隐私")