"""Streamlit frontend for the Medical AI Assistant.

Calls the FastAPI backend at http://localhost:8000.
Run the API first: uvicorn src.api.app:app --reload --port 8000
Then run this: streamlit run app.py --server.port 8501
"""

import os
from typing import Any

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")


def _api_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """Make an API request to the FastAPI backend."""
    url = f"{API_BASE}{path}"
    try:
        resp = requests.request(method, url, timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(f"无法连接到后端服务: {url}")
        return {}
    except requests.Timeout:
        st.error("请求超时，请稍后重试")
        return {}


# ============================================================
# Sidebar — conversation list + actions
# ============================================================
with st.sidebar:
    st.title("🏥 医疗设备知识助手")
    st.markdown("---")

    # --- Load conversation list ---
    if "conversation_list" not in st.session_state:
        st.session_state["conversation_list"] = None

    def refresh_conv_list():
        data = _api_request("GET", "/api/conversations?page=1&page_size=50")
        # Filter out conversations with 0 messages
        convs = [c for c in data.get("conversations", []) if c["message_count"] > 0]
        st.session_state["conversation_list"] = convs

    refresh_conv_list()

    # --- New conversation button ---
    if st.button("➕ 新建对话", use_container_width=True):
        conv = _api_request("POST", "/api/conversations", json={"title": "新对话"})
        if conv and conv.get("id"):
            st.session_state["conversation_id"] = conv["id"]
            st.session_state["messages"] = []
            st.session_state["selected_conv_index"] = None
            refresh_conv_list()
            st.rerun()

    # --- Conversation list ---
    if st.session_state["conversation_list"]:
        st.markdown("**历史对话**")
        for idx, conv in enumerate(st.session_state["conversation_list"]):
            btn_label = f"💬 {conv['title']} ({conv['message_count']}条)"
            if conv["id"] == st.session_state.get("conversation_id"):
                btn_label = f"🔵 {conv['title']} ({conv['message_count']}条)"

            col_select, col_del = st.columns([9, 1])
            if col_select.button(btn_label, key=f"conv_btn_{idx}", use_container_width=True):
                st.session_state["conversation_id"] = conv["id"]
                # Load message history from backend
                history_resp = _api_request("GET", f"/api/conversations/{conv['id']}/messages")
                msgs = history_resp.get("messages", []) if history_resp else []
                st.session_state["messages"] = [
                    {"role": m["role"], "content": m["content"], "sources": _parse_sources(m.get("sources", ""))}
                    for m in msgs
                ]
                st.session_state["selected_conv_index"] = idx
                st.rerun()

            if col_del.button("🗑️", key=f"del_btn_{idx}"):
                _api_request("DELETE", f"/api/conversations/{conv['id']}")
                if st.session_state.get("conversation_id") == conv["id"]:
                    st.session_state.pop("conversation_id", None)
                    st.session_state["messages"] = []
                refresh_conv_list()
                st.rerun()
    else:
        # No conversations with messages — auto-create a fresh one
        conv = _api_request("POST", "/api/conversations", json={"title": "新对话"})
        if conv and conv.get("id"):
            st.session_state["conversation_id"] = conv["id"]
            st.session_state["messages"] = []
            refresh_conv_list()

    st.markdown("---")
    conv_id = st.session_state.get("conversation_id")
    if conv_id:
        st.caption(f"当前: {conv_id[:8]}...")
    st.caption(f"Backend: {API_BASE}")


def _parse_sources(sources_str: str) -> list[dict]:
    """Parse JSON sources string from DB into a list of dicts."""
    if not sources_str:
        return []
    try:
        import json
        return json.loads(sources_str)
    except Exception:
        return []


# ============================================================
# Auto-create conversation if none selected
# ============================================================
if "conversation_id" not in st.session_state or not st.session_state["conversation_id"]:
    conv = _api_request("POST", "/api/conversations", json={"title": "新对话"})
    if conv and conv.get("id"):
        st.session_state["conversation_id"] = conv["id"]
    else:
        st.session_state["conversation_id"] = "local-demo"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ============================================================
# Render chat history
# ============================================================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 参考资料"):
                for s in msg["sources"][:3]:
                    st.markdown(f"**[{s.get('index', '?')}]** {s.get('source', '')}: {s.get('content', '')[:150]}...")

# ============================================================
# User input & API call
# ============================================================
prompt = st.chat_input("请输入您的医疗设备问题...")

if prompt:
    # Show user message
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("AI 正在思考..."):
            resp = _api_request(
                "POST",
                f"/api/conversations/{st.session_state['conversation_id']}/messages",
                json={"message": prompt},
            )

    reply = resp.get("reply", "抱歉，未能获取回复。")
    sources = resp.get("sources", [])

    st.write(reply)
    if sources:
        with st.expander("📚 参考资料"):
            for s in sources:
                st.markdown(f"**[{s.get('index', '?')}]** {s.get('source', '')}: {s.get('content', '')[:200]}")

    st.session_state["messages"].append({
        "role": "assistant",
        "content": reply,
        "sources": sources,
    })
    st.rerun()
