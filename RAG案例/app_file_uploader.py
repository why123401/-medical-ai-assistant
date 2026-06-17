import time

import streamlit  as st
from knowledge_base import KnowledgeBaseService

#添加网页标题

st.title("知识库更新服务")

uploader_file=st.file_uploader(
    "请上传TXT文件",
    type=['txt'],
    accept_multiple_files=False,


)
if "service" not in st.session_state:
    st.session_state["service"]=KnowledgeBaseService()
if uploader_file is not None:
    file_name=uploader_file.name
    file_type = uploader_file.type
    file_size = uploader_file.size/1024

    st.subheader(f"文件名:{file_name}")
    st.write(f"大小:{file_size:.2f}KB|格式:{file_type}")

    text=uploader_file.getvalue().decode("utf-8")
    with st.spinner("载入知识库中..."):
        time.sleep(1)

        result=st.session_state["service"].upload_by_str(text, file_name)
        st.write(result)