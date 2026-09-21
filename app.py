import os
import re
import json
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 1. CONFIGURATION ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="UniChalo AI", page_icon="🎓")
st.title("🎓 UniChalo AI Assistant")
st.caption("Ask me anything about university admissions! (Powered by Qwen & Groq Cloud)")

@st.cache_resource
def load_knowledge():
    try:
        with open("knowledge.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

knowledge_db = load_knowledge()

def chunk_text(text, chunk_size_lines=20, overlap=4):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    chunks = []
    i = 0
    while i < len(lines):
        chunk_lines = lines[i:i + chunk_size_lines]
        chunks.append("\n".join(chunk_lines))
        i += chunk_size_lines - overlap
    return chunks

def retrieve_relevant_context(query, knowledge):
    query_lower = query.lower()
    
    mapping = {
        'ned': 'ned',
        'neduet': 'ned',
        'fast': 'fast',
        'nu': 'fast',
        'nuces': 'fast',
        'dow': 'dow',
        'duhs': 'dow',
        'jsmu': 'jsmu',
        'sindh medical': 'jsmu',
        'kmu': 'kmu',
        'kmdc': 'kmu',
        'khyber': 'kmu',
        'smbbmc': 'smbbmc',
        'benazir': 'smbbmc',
        'lyari': 'smbbmc',
        'karachi': 'karachi',
        'ku': 'karachi',
        'uok': 'karachi',
        'sir syed': 'sir syed',
        'ssuet': 'sir syed',
        'dawood': 'dawood',
        'duet': 'dawood',
        'ubit': 'ubit',
        'umaer basha': 'ubit',
        'dcs uok': 'ubit',
        'iba': 'iba',
        'aku': 'aku',
        'aga khan': 'aku',
        'agakhan': 'aku'
    }
    
    matched_keys = set()
    for keyword, key in mapping.items():
        if re.search(rf'\b{re.escape(keyword)}\b', query_lower):
            matched_keys.add(key)
            
    if not matched_keys:
        return "NO_UNI_DETECTED"
    
    query_words = set(re.findall(r'\w+', query_lower))
    stop_words = {'what', 'is', 'the', 'a', 'an', 'for', 'in', 'of', 'to', 'and', 'how', 'are', 'you', 'university', 'tell', 'me', 'about', 'can', 'please'}
    query_words = query_words - stop_words
    
    selected_contexts = []
    for uni_key in matched_keys:
        uni_text = knowledge.get(uni_key, "")
        if not uni_text:
            continue
            
        chunks = chunk_text(uni_text, chunk_size_lines=20, overlap=4)
        
        scores = []
        for c in chunks:
            c_lower = c.lower()
            c_words = set(re.findall(r'\w+', c_lower))
            score = sum(3 for w in query_words if w in c_words)
            for w in query_words:
                if w in c_lower:
                    score += 1
            scores.append((score, c))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        best_chunks = [c for s, c in scores[:3] if s > 0]
        if not best_chunks:
            best_chunks = chunks[:2]
            
        uni_content = "\n\n---\n\n".join(best_chunks)
        selected_contexts.append(f"--- Information about {uni_key.upper()} University ---\n{uni_content}")
        
    return "\n\n====================\n\n".join(selected_contexts)

@st.cache_resource
def init_llm_chain():
    # max_retries handles transient cloud network blips right after waking up
    # max_tokens=600 ensures requests never exceed Groq's output tokens per minute (OTPM) rate limit
    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, max_tokens=600, max_retries=3, request_timeout=30)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are the UniChalo AI admission guide, an expert assistant for university admissions in Pakistan "
            "(including NED, FAST, DUHS/Dow, Karachi University, Dawood, SSUET, JSMU, KMU, SMBBMC, UBIT, IBA, AKU, etc.).\n\n"
            "INSTRUCTIONS:\n"
            "1. Answer clearly, accurately, and concisely based ONLY on the provided Context below.\n"
            "2. If the context does not contain the answer, politely state that you do not have that specific information in your records.\n"
            "3. If the context says 'NO_UNI_DETECTED', greet the user warmly and ask which university they would like information about.\n"
            "4. Never hallucinate or mix up details between different universities.\n"
            "5. If asked about your identity, ALWAYS state you are the 'UniChalo AI Assistant'. Do NOT mention Groq, Qwen, Alibaba, or underlying models.\n\n"
            "Context:\n{context}"
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    return qa_prompt | llm

llm_chain = init_llm_chain()

# Each browser session gets its own unique history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

def invoke_with_context(user_input):
    context = retrieve_relevant_context(user_input, knowledge_db)
    
    # Trim chat history to last 6 messages to prevent context ballooning and rate limit overruns
    if len(st.session_state.chat_history.messages) > 6:
        st.session_state.chat_history.messages = st.session_state.chat_history.messages[-6:]
        
    chain_with_history = RunnableWithMessageHistory(
        llm_chain,
        lambda session_id: st.session_state.chat_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    # Retry loop with backoff in case of wake-up cold start or rate spikes
    last_err = None
    for attempt in range(3):
        try:
            response = chain_with_history.invoke(
                {"input": user_input, "context": context},
                config={"configurable": {"session_id": "session_user"}}
            )
            return response.content
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "rate" in err_str or "status" in err_str or "timeout" in err_str or "connection" in err_str:
                time.sleep(2 * (attempt + 1))
            else:
                break
                
    return f"⚠️ The AI service is currently warming up or reaching its connection limit. Please try asking again in a few seconds! (Details: {last_err})"

# --- WEB INTERFACE (STREAMLIT) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text = invoke_with_context(prompt)
            st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
