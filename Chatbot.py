import streamlit as st
from logic import get_response_stream

st.title("Chatbot")
st.caption("Streamlit chatbot with Ollama")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    response_stream = get_response_stream(prompt)

    def stream_text():
        for chunk in response_stream:
            yield str(chunk)

    with st.chat_message("assistant"):
        msg = st.write_stream(stream_text())

    st.session_state.messages.append({"role": "assistant", "content": msg})
