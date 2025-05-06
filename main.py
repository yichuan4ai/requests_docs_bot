import os

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.chains import RetrievalQA

# 设置USER_AGENT（必须在导入WebBaseLoader之前）
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 定义要加载的文档 URL 列表
# 这里我们选择 requests 文档的几个核心页面
urls = [
    "https://requests.readthedocs.io/en/latest/",  # 首页/快速开始
    "https://requests.readthedocs.io/en/latest/user/quickstart/",  # 快速开始
    "https://requests.readthedocs.io/en/latest/user/advanced/",  # 高级用法
    "https://requests.readthedocs.io/en/latest/api/"  # API 参考 (部分)
]

# 使用 WebBaseLoader 加载文档
loader = WebBaseLoader(urls)
docs = loader.load()

print(f"成功加载了 {len(docs)} 个文档页面。")
# 可以打印第一个文档的内容看看
# print(docs[0].page_content[:500])


# 初始化文本分割器
# chunk_size: 每个文本块的最大字符数
# chunk_overlap: 相邻文本块之间的重叠字符数
# 对于技术文档，chunk_size 不宜过小，以保留代码块或段落的完整性
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# 分割文档
splits = text_splitter.split_documents(docs)

print(f"原始文档分割成了 {len(splits)} 个文本块。")
# 可以打印第一个分割块的内容看看
# print(splits[0].page_content)


# embedding
model_name = "BAAI/bge-small-en-v1.5"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# # 初始化向量数据库并从分割后的文本块创建
# # directory 参数指定向量数据存储的路径，方便后续加载
persist_directory = './chroma_db'
if os.path.exists(persist_directory):
    # 如果数据库已经存在，下次可以直接加载
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print(f"文本块已嵌入并存储到向量数据库：{persist_directory}")
else:
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
    print(f"已加载向量数据库：{persist_directory}")


# 从向量数据库创建检索器
# search_kwargs={"k": 3} 表示检索最相似的 3 个文本块
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print(f"已创建检索器，每次查询将返回最相似的 {retriever.search_kwargs['k']} 个文本块。")

# # 可以测试一下检索效果
# query = "How to set a custom header in requests?"
# retrieved_docs = retriever.invoke(query)
# print(f"\n对查询 '{query}' 检索到 {len(retrieved_docs)} 个文档：")
# for i, doc in enumerate(retrieved_docs):
#     print(f"--- 文档 {i+1} (来源: {doc.metadata.get('source', '未知')}) ---")
#     print(doc.page_content[:200] + "...") # 打印前200字符
#


# 提出问题
query = "How to set a custom header in a requests GET request?"
# query = "What is the timeout parameter used for?"
# query = "How can I upload a file using requests?"

print(f"\n--- 提问: {query} ---")

deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
# 初始化大型语言模型
# temperature=0 表示希望模型回答更确定、更少创造性，适合问答任务
llm = ChatDeepSeek(model="deepseek-chat", api_key=deepseek_api_key, temperature=0)

# 创建 RetrievalQA Chain
# retriever: 使用我们之前创建的检索器
# llm: 使用我们初始化的 LLM
# return_source_documents=True: 设置为 True 可以让 Chain 返回检索到的原始文档，方便验证
qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff", # "stuff" 链类型将所有检索到的文档填充到 LLM 的上下文
    retriever=retriever,
    return_source_documents=True
)

print("已成功组装 RetrievalQA Chain。")
# 运行 Chain 获取答案
response = qa_chain.invoke({"query": query})

# 打印答案
print("\n--- 答案 ---")
print(response["result"])

# 打印检索到的原始文档（因为 return_source_documents=True）
print("\n--- 答案来源 ---")
if "source_documents" in response:
    for i, doc in enumerate(response["source_documents"]):
        print(f"文档 {i+1} (来源: {doc.metadata.get('source', '未知')})")
        # print(doc.page_content[:300] + "...") # 可以打印部分内容查看
else:
    print("未返回来源文档。")