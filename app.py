import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 1. CONFIGURATION ---
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="UniChalo AI", page_icon="🎓")
st.title("🎓 UniChalo AI Assistant")
st.caption("Ask me anything about university admissions! (Powered by Qwen & Groq Cloud)")

@st.cache_resource
def load_knowledge_base():
    try:
        with open("knowledge_base.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "No knowledge base found. Please add knowledge_base.txt"

knowledge_context = load_knowledge_base()

@st.cache_resource
def init_llm_chain():
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0) # Using fast, highly available model with high token limits
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are the UniChalo AI admission guide, a comprehensive assistant for MULTIPLE universities (including NED, FAST, IBA, KU, Dow, etc.). NEVER claim to be exclusively for one university. Use the following context to answer the user's question accurately. If the context doesn't contain the answer, just say you don't know. If asked about your identity, creator, or model, ALWAYS say 'I am the UniChalo AI Assistant.' Do NOT mention Alibaba, Qwen, Tongyi Lab, or Groq.\n\nContext:\n{knowledge_context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    chain = qa_prompt | llm
    return chain

llm_chain = init_llm_chain()

# Use Streamlit's session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

conversational_chain = RunnableWithMessageHistory(
    llm_chain,
    lambda session_id: st.session_state.chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# --- WEB INTERFACE (STREAMLIT) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Draw all past messages on the screen
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User Input Box
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        response = conversational_chain.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": "streamlit_cloud_session"}}
        )
        st.write(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})

