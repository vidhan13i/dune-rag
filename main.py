import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# IMPORT YOUR FUNCTIONS
from app import load_embeddings_and_db, load_llm, create_conversational_chain

# ---- PAGE CONFIG ----
st.set_page_config(page_title="DUNE RAG", layout="wide")

st.title("🏜️ DUNE RAG Chatbot")
st.markdown("Ask anything about the Dune universe")


# ---- LOAD MODELS (CACHE) ----
@st.cache_resource
def initialize():
    db = load_embeddings_and_db()
    llm = load_llm()
    chain = create_conversational_chain(db, llm)
    return chain


chain = initialize()

# ---- SESSION STATE ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.info("Dune RAG using FAISS + Groq + MiniLM embeddings")

# ---- DISPLAY CHAT ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- USER INPUT ----
query = st.chat_input("Ask about Dune...")

if query:
    # show user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain({
                "question": query,
                "chat_history": st.session_state.chat_history
            })

            answer_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            st.markdown(answer_text)

    # update memory
    st.session_state.chat_history.append(HumanMessage(content=query))
    st.session_state.chat_history.append(AIMessage(content=answer_text))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text
    })