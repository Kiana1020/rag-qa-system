"""
文档加载与文本分割模块
支持 .txt, .pdf, .docx 格式
"""
import os
from typing import List

# LangChain 的文档加载器
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    遍历文件夹，根据文件扩展名选择对应的加载器，返回 Document 对象列表
    """
    all_docs = []
    supported_ext = ['.txt', '.pdf', '.docx']

    for file_name in os.listdir(folder_path):
        file_ext = os.path.splitext(file_name)[1].lower()
        if file_ext not in supported_ext:
            print(f"跳过不支持的文件格式: {file_name}")
            continue

        file_path = os.path.join(folder_path, file_name)
        print(f"正在加载: {file_path}")

        try:
            if file_ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.docx':
                loader = Docx2txtLoader(file_path)
            else:
                continue

            docs = loader.load()
            # 为每个文档添加来源信息（方便调试）
            for doc in docs:
                doc.metadata["source"] = file_name
            all_docs.extend(docs)
        except Exception as e:
            print(f"加载文件 {file_name} 时出错: {e}")

    print(f"共加载了 {len(all_docs)} 个文档块（每个PDF的一页或每个txt/docx整体视为一个Document）")
    return all_docs


def split_documents(docs: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    将文档切分成更小的块，以便于检索。
    chunk_size: 每个块的最大字符数（不是 token 数，简化处理）
    chunk_overlap: 块之间的重叠字符数，保持上下文连贯
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"切分后共得到 {len(split_docs)} 个文本块")
    return split_docs


# 简单的测试：当直接运行此脚本时，演示加载和切分过程
if __name__ == "__main__":
    # 设置 data 文件夹路径（相对于项目根目录）
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    if not os.path.exists(data_folder):
        print("警告: data 文件夹不存在，请先创建并在其中放入测试文档。")
    else:
        docs = load_documents_from_folder(data_folder)
        if docs:
            chunks = split_documents(docs)
            # 打印前两个块的内容预览
            for i, chunk in enumerate(chunks[:2]):
                print(f"\n--- 块 {i + 1} 预览 ---")
                print(chunk.page_content[:200])  # 只显示前200字符