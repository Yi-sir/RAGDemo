# RAG Demo

简易的RAG Demo，适合个人/组织部署使用。

## 特性

* 支持`API`调用或本地调用大模型
* 本地调用大模型支持`vllm`或`transformers`库
* 支持多用户访问

更多功能持续开发中，欢迎一起建设！

## 使用方法

```shell
# 注意，默认的配置文件使用api调用llm，使用本地权重调用embedding model。使用前可以根据实际状况修改
pip install -r requirements.txt
streamlit run ./app/st_demo.py
```

## 参考配置

配置文件：[config.json](app/config/config.json)

```json
{
    "llm_config": {
        "model": "spn3/Qwen2.5-72B-Instruct",
        "backend_type": "Api",
        "path": "",
        "api_url": "https://www.sophnet.com/api/open-apis",
        "api_key": ""
    },
    "doc_config": {
        "embedding_model": "bge",
        "embedding_model_path": "./bge-small-zh",
        "split_method": "FixedLength",
        "chunk_length": 512,
        "overlap": 50,
        "database_method": "Faiss",
        "dimension": 512,
        "topk": 1
    }
}
```

## 可选参数列表

| 参数名称 | 可选项 | 含义 |
| ------ | ----- | ----- |
| model | - | 模型名称 |
| backend_type | "api", "local" | `llm`推理方式，"api"为调用`api`推理，"local"为本地推理，暂时支持`vllm`、`transformers` |
| path | - | 本地模型权重文件路径。当`backend_type`为`local`时，与`model`参数拼接送给推理后端 |
| api_url | - | Api url。当`backend_type`为`api`时，用于初始化`client` |
| api_key | - | Api key。当`backend_type`为`api`时，用户初始化`client`。会优先从环境变量中读取`RAG_GENERATOR_API_KEY`，若读不到则读取配置文件 |
| embedding_model | "bge" | `embedding`模型名称 |
| embedding_model_path | - | `embedding`模型路径。如果模型路径存在则从本地初始化，否则根据模型名称从云端仓库拉取 |
| split_method | "fixedlength" | 文档切分逻辑，"fixedlength"指按照固定长度切分 |
| chunk_length | - | 当`split_method`为`fixedlength`时，切分出每个文本块的长度 |
| overlap | - | 当`split_method`为`fixedlength`时，切分文本块时，相邻块之间重叠的长度 |
| database_method | "faiss" | `embedding`向量搜索的方式 |
| dimension | - | `embedding`向量长度 |
| topk | - | 搜索相关文本块时结果的最大数量 |

## 说明

> 对于本地部署`VLLM`、`TGI`等`LLM`框架，本项目通过API调用推理接口的情况：`backend_type`设置为`api`，`api_url`设置为`http://localhost:port/v1`，`api_key`置空即可。

> 本项目依赖`streamlit`搭建前端页面，而`streamlit`对多用户数据实现了隔离，所以理论上可以在云端部署，提供给一定规模的组织使用。