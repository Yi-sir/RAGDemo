# git clone https://github.com/FlagOpen/FlagEmbedding.git
# cd FlagEmbedding && pip3 install -e ./
# git clone https://www.modelscope.cn/BAAI/bge-small-zh.git   # for example

import time

from FlagEmbedding import FlagAutoModel

model = FlagAutoModel.from_finetuned(
    "/workspace/RAG/bge-small-zh",
    query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
    use_fp16=True,
)

sentences_1 = ["I love NLP", "I love machine learning"]
sentences_2 = ["I love BGE", "I love text retrieval"]


sentence_3 = [
    "10/22/2024：我们发布了新的模型：OmniGen，这是一个支持各种任务的统一图像生成模型。OmniGen可以在不需要额外插件（如ControlNet、IP-Adapter）或辅助模型（如姿态检测和人脸检测）的情况下完成复杂的图像生成任务。 \
9/10/2024：我们推出了MemoRAG，这是一种基于记忆启发的知识发现技术，是迈向 RAG 2.0 的关键一步（仓库：https://github.com/qhjqhj00/MemoRAG，论文：https://arxiv.org/pdf/2409.05591v1） 🔥 \
9/2/2024: 开始维护更新教程，教程文件夹中的内容会在未来不断丰富，欢迎持续关注！ 📚 \
7/26/2024：发布bge-en-icl。这是一个结合了上下文学习能力的文本检索模型，通过提供与任务相关的查询-回答示例，可以编码语义更丰富的查询，进一步增强嵌入的语义表征能力。 🔥 \
7/26/2024: 发布bge-multilingual-gemma2。这是一个基于gemma-2-9b的多语言文本向量模型，同时支持多种语言和多样的下游任务，在多语言检索数据集 MIRACL, MTEB-fr, MTEB-pl 上取得了迄今最好的实验结果。 🔥 \
7/26/2024：发布新的轻量级重排器bge-reranker-v2.5-gemma2-lightweight。这是一个基于gemma-2-9b的轻量级重排器，支持令牌压缩和分层轻量操作，在节省大量资源的同时，仍能确保良好的性能。:fire:"
]


start_time = time.time()
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
embeddings_3 = model.encode(sentence_3)
end_time = time.time()


print(f"==== embeddings_1: {embeddings_1} ====")
print(f"==== embeddings_2: {embeddings_2} ====")
print(f"==== embeddings_3: {embeddings_3} ====")

print(f"==== time is {end_time - start_time} ====")
