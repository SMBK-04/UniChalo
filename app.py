import os
import re
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
def load_and_chunk_knowledge_base():
    try:
        with open("knowledge_base.txt", "r", encoding="utf-8") as f:
            text = f.read()
            # Split by double newlines to create natural paragraph chunks
            raw_chunks = text.split("\n\n")
            # Group chunks so they are roughly 500-1000 characters each
            chunks = []
            current_chunk = ""
            for rc in raw_chunks:
                current_chunk += rc + "\n"
                if len(current_chunk) > 800:
                    chunks.append(current_chunk)
                    current_chunk = ""
            if current_chunk:
                chunks.append(current_chunk)
            return chunks
    except FileNotFoundError:
        return []

knowledge_chunks = load_and_chunk_knowledge_base()

def retrieve_relevant_context(query, chunks, top_k=8):
    # Pure Python Keyword Search (0 RAM, 0 extra cost)
    query_words = set(re.findall(r'\w+', query.lower()))
    
    # Ignore common stop words
    stop_words = {"what", "is", "the", "a", "an", "for", "in", "of", "to", "and", "how", "are", "you"}
    query_words = query_words - stop_words
    
    if not query_words or not chunks:
        return "General knowledge context."
        
    scores = []
    for chunk in chunks:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = sum(1 for word in query_words if word in chunk_words)
        scores.append((score, chunk))
    
    # Sort descending by score
    scores.sort(key=lambda x: x[0], reverse=True)
    best_chunks = [chunk for score, chunk in scores[:top_k] if score > 0]
    
    return "\n\n---\n\n".join(best_chunks)

@st.cache_resource
def init_llm_chain():
    # Use qwen/qwen3.8-27b as the flagship model
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the UniChalo AI admission guide, a comprehensive assistant for MULTIPLE universities (including NED, FAST, IBA, KU, Dow, etc.). NEVER claim to be exclusively for one university. Use the following context to answer the user's question accurately. If the context doesn't contain the answer, just say you don't know. If asked about your identity, creator, or model, ALWAYS say 'I am the UniChalo AI Assistant.' Do NOT mention Alibaba, Qwen, Tongyi Lab, or Groq.\n\nContext:\n{context}"),
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
    context = retrieve_relevant_context(user_input, knowledge_chunks)
    
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
