# 智能文档问答助手 (RAG Application)

基于 **LangChain + FAISS + Ollama + Streamlit** 构建的本地化智能问答系统。支持上传 `.txt`、`.pdf`、`.docx` 文档，通过检索增强生成（RAG）技术，让大语言模型基于你的私有文档回答问题，避免胡编乱造。

## ✨ 功能特点

- 📄 支持多种文档格式（TXT、PDF、DOCX）
- 🔍 本地向量检索（FAISS + 多语言 Embedding 模型）
- 💬 基于 Ollama 的本地大模型推理（完全免费，无需 GPU）
- 🌐 简洁的 Web 界面（Streamlit）
- 📌 答案引用来源，可追溯
- 🚀 命令行交互模式 + 可视化界面双支持

## 🛠️ 技术栈

- **框架**: LangChain 0.2.x
- **向量数据库**: FAISS (CPU 版)
- **嵌入模型**: `paraphrase-multilingual-MiniLM-L12-v2` (支持 50+ 种语言，中文友好)
- **大语言模型**: Ollama + Qwen2-1.5B (可替换为 Llama3.2、Phi 等)
- **前端**: Streamlit
- **文档解析**: pypdf, python-docx

## 📸 效果预览


> `![问答示例](images/demo.png)`

## 🚀 快速开始

### 环境要求

- Python 3.10 ~ 3.12
- [Ollama](https://ollama.com/download/windows) (已安装并下载至少一个模型，例如 `qwen2:1.5b`)

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/你的用户名/rag_qa_system.git
   cd rag_qa_system
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. **准备文档**  
   将你的 `.txt`、`.pdf`、`.docx` 文件放入 `data/` 文件夹（如果没有则新建）。

5. **构建向量索引**（首次运行会自动下载嵌入模型，约 500MB）
   ```bash
   python src/build_index.py
   ```

6. **启动 Streamlit 界面**
   ```bash
   streamlit run src/app.py
   ```
   浏览器打开 `http://localhost:8501` 即可使用。

### 命令行问答（备选）
```bash
python src/qa_chain.py
```

## 📂 项目结构

```
rag_qa_system/
├── data/                      # 存放待索引的文档（请勿上传敏感文件到GitHub）
├── vector_store/              # 持久化的FAISS向量索引（自动生成，已gitignore）
├── src/                       # 核心源码
│   ├── __init__.py
│   ├── load_docs.py           # 文档加载与文本分割
│   ├── build_index.py         # 构建向量数据库
│   ├── qa_chain.py            # 问答链逻辑（命令行版）
│   └── app.py                 # Streamlit Web界面
├── .gitignore                 # 忽略临时文件
├── requirements.txt           # 项目依赖
├── README.md                  # 项目说明
└── .env                       # 环境变量（可选）
```

## 🎯 使用示例

**提问**：“人工智能有哪些主要分支？”  
**回答**：系统将从你上传的文档中检索相关内容，生成类似下面的答案：

> 根据文档《AI简介.txt》，人工智能主要包括机器学习、计算机视觉、自然语言处理和机器人学等分支。  
> *（同时显示引用来源）*

## 🧠 自定义模型

1. 通过 Ollama 安装其他模型（例如 `llama3.2:3b`）：
   ```bash
   ollama pull llama3.2:3b
   ```
2. 修改 `src/qa_chain.py` 和 `src/app.py` 中的 `model_name` 参数为你下载的模型名称。

## 📝 常见问题

**Q: 构建索引时下载嵌入模型失败怎么办？**  
A: 可以设置国内镜像源，或手动下载模型文件夹放到 `~/.cache/huggingface/hub/` 目录下。

**Q: 问答速度慢怎么办？**  
A: 使用更小的模型如 `qwen2:0.5b`，或减少 `chunk_size` 和检索块数 `k`。

**Q: 模型回答仍然瞎编？**  
A: 调低 `temperature`（例如 0.0），并确认文档中确实包含相关信息。

## 🚧 未来改进计划

- [ ] 支持更多文档格式（Markdown、HTML）
- [ ] 添加对话历史记忆功能
- [ ] 优化分块策略（语义分块、滑动窗口）
- [ ] 部署到 Hugging Face Spaces（需替换 Ollama 为在线 API）

## 📄 License

MIT License  
Copyright (c) 2025 你的名字

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/)