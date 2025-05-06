# install dependencies

## 初始化项目环境

```bash
uv venv
source .venv/bin/activate
uv add langchain langchain-community langchain_huggingface  bs4 python-dotenv
uv add text2vec transformers[torch] sentence-transformers chromadb
```