import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from app.ingestion import (
    load_and_index,
    get_vectorstore,
    set_vectorstore,
    load_vectorstore,
)
from app.rag_chain import get_rag_chain

load_dotenv()

st.set_page_config(page_title="📄 Document Chatbot", layout="wide")
st.title("📚 Ask Questions About Your Documents")

# Store chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: LLM selector (only OpenAI if you want)
st.sidebar.subheader("🧠 Choose a Language Model")
st.session_state.model = st.sidebar.selectbox("LLM Backend", ["OpenAI"])  # simplified to only OpenAI

# Load existing vectorstore once (OpenAI embeddings only)
if get_vectorstore() is None:
    vs = load_vectorstore()  # no embedding_model param here
    if vs:
        set_vectorstore(vs)

# File upload section
st.subheader("Upload a Document")
uploaded_file = st.file_uploader("Choose a document (.pdf, .txt, .md, .docx)", type=["pdf", "txt", "md", "docx"])

if uploaded_file is not None:
    file_path = Path(f"uploaded_files/{uploaded_file.name}")
    file_path.parent.mkdir(exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    load_and_index(file_path)  # no embedding_model param
    st.success(f"✅ Uploaded and indexed: {uploaded_file.name}")

# Chat input and response
st.subheader("💬 Chat with your document")

user_input = st.chat_input("Ask a question...")
if user_input:
    try:
        # Reload vectorstore from disk (OpenAI embeddings only)
        vs = load_vectorstore()
        if vs:
            set_vectorstore(vs)

        chain = get_rag_chain(model_name=st.session_state.model)
        answer = chain.run(user_input)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("ai", answer))

    except ValueError as e:
        st.error(str(e))

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"**🧑‍💻 You:** {msg}")
    else:
        st.markdown(f"**🤖 Assistant:** {msg}")
