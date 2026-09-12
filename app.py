import os
import re
import json
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 1. CONFIGURATION ---
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="UniChalo AI", page_icon="🎓")
st.title("🎓 UniChalo AI Assistant")
st.caption("Ask me anything about university admissions! (Powered by Qwen & Groq Cloud)")

@st.cache_resource
def load_knowledge_json():
    try:
        with open("knowledge.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

knowledge_db = load_knowledge_json()

def retrieve_relevant_context(query, knowledge):
    query = query.lower()
    
    # Map keywords to dictionary keys
    mapping = {
        'ned': 'ned',
        'fast': 'fast',
        'dow': 'dow',
        'duhs': 'dow',
        'jsmu': 'jsmu',
        'sindh medical': 'jsmu',
        'kmu': 'kmu',
        'khyber': 'kmu',
        'smbbmc': 'smbbmc',
        'benazir': 'smbbmc',
        'lyari': 'smbbmc',
        'karachi': 'karachi',
        'ku': 'karachi',
        'sir syed': 'sir syed',
        'ssuet': 'sir syed',
        'dawood': 'dawood',
        'duet': 'dawood'
    }
    
    matched_keys = set()
    for keyword, key in mapping.items():
        # Use regex to match exact words (prevent matching 'ned' inside 'happened')
        if re.search(rf'\b{keyword}\b', query):
            matched_keys.add(key)
            
    # If a specific university is detected, return ONLY its full context!
    if matched_keys:
        contexts = []
        for key in matched_keys:
            contexts.append(f"--- Information about {key.upper()} University ---\n" + knowledge.get(key, ""))
        return "\n\n".join(contexts)
    
    # If no university is detected, provide a generic prompt for the LLM
    return "The user hasn't specified a university. Ask them which university they are interested in (e.g., NED, FAST, Karachi University, Dow, etc.)."

@st.cache_resource
def init_llm_chain():
    # Use qwen/qwen3.8-27b as the flagship model
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the UniChalo AI admission guide, a comprehensive assistant for MULTIPLE universities. Use the following context to answer the user's question accurately. If the context says 'ask them which university', do exactly that. If the context doesn't contain the answer, just say you don't know based on the provided documents. If asked about your identity, creator, or model, ALWAYS say 'I am the UniChalo AI Assistant.' Do NOT mention Alibaba, Qwen, Tongyi Lab, or Groq.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    chain = qa_prompt | llm
    return chain

llm_chain = init_llm_chain()

# Use Streamlit's session memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

# Custom Runnable that fetches context before passing to LLM
def invoke_with_context(user_input):
    context = retrieve_relevant_context(user_input, knowledge_db)
    
    chain_with_history = RunnableWithMessageHistory(
        llm_chain,
        lambda session_id: st.session_state.chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    response = chain_with_history.invoke(
        {"input": user_input, "context": context},
        config={"configurable": {"session_id": "streamlit_cloud_session"}}
    )
    return response.content

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
        response_text = invoke_with_context(prompt)
        st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
