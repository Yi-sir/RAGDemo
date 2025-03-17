import os
import sys

import streamlit as st
from utils.logger import setup_logging

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.path.join(BASE_DIR, "app/config")
RAG_ENGINE_CONFIG_PATH = os.path.join(JSON_DIR, "config.json")
RAG_LOGGER_CONFIG_PATH = os.path.join(JSON_DIR, "logging.json")

sys.path.append(BASE_DIR)
setup_logging(RAG_LOGGER_CONFIG_PATH)

from app.engine.config import RAGConfig
from app.engine.rag_engine import RAGEngine

if __name__ == "__main__":

    # 页面标题
    st.title("RAG Demo")

    # 初始化会话状态
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = RAGEngine(
            config=RAGConfig.from_json(RAG_ENGINE_CONFIG_PATH)
        )

    # 文件上传功能
    st.header("选择文档")
    uploaded_file = st.file_uploader(
        "选择一个文档（支持 PDF、DOCX、TXT 等格式）", type=["pdf", "docx", "txt"]
    )
    submitted = st.button("上传")
    if uploaded_file is not None and submitted:
        # 保存上传的文件到临时路径
        file_path = f"/tmp/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 添加文档到 RAG 系统
        if st.session_state.rag_engine.add_doc(file_path):
            st.success(f"文档 '{uploaded_file.name}' 添加成功！")
        else:
            st.error(f"文档 '{uploaded_file.name}' 添加失败！")

    # 问答功能
    st.header("问答")
    question = st.text_input("请输入您的问题：")
    if st.button("提交"):
        if question:
            with st.spinner("正在生成答案..."):
                result = st.session_state.rag_engine.query(question)
                if result["answer"]:
                    st.success("答案：")
                    st.write(result["answer"])
                    if result["reference"]:
                        st.info("参考文档：")
                        for ref in result["reference"]:
                            st.write(ref)
                else:
                    st.error("无法生成答案，请稍后重试。")
        else:
            st.warning("请输入问题！")

    # 显示系统状态
    st.header("系统状态")
    status = st.session_state.rag_engine.get_status()
    st.json(status)

    # 删除文档功能
    st.header("删除文档")
    document_list = st.session_state.rag_engine.get_doc_list()
    if document_list:
        selected_document = st.selectbox("选择要删除的文档", document_list)
        if st.button("删除"):
            if st.session_state.rag_engine.remove_doc(selected_document):
                st.success(f"文档 '{selected_document}' 删除成功！")
            else:
                st.error(f"文档 '{selected_document}' 删除失败！")
    else:
        st.info("当前没有加载的文档。")
