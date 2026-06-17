import time

import streamlit as st

from agent.react_agent import ReactAgent

# 标题
st.title("智扫通机器人智能客服")
st.divider()

# 初始化 session_state
if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()

if "message" not in st.session_state:
    st.session_state["message"] = []

# 渲染历史消息
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    # 1. 显示用户消息
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    # 2. 准备调用 Agent
    response_messages = []

    with st.spinner("智能客服思考..."):
        res_stream = st.session_state["agent"].execute_stream(prompt)


    # 定义 capture 函数（如果 ReactAgent 内部没做流式处理，可以在这里包裹）

    def capture(generator, cache_list):
        for chunk in generator:
            cache_list.append(chunk)

            for char in chunk:
                time.sleep(0.01)
                yield char


    st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
    st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
    st.rerun()
