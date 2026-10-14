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

def load_knowledge():
    try:
        with open("knowledge.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

knowledge_db = load_knowledge()

def extract_sections(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    sections = []
    curr_title = "Overview"
    curr_lines = []
    
    for line in lines:
        if line.startswith("Source:"):
            continue
        is_heading = (
            (bool(re.match(r'^(SECTION\s+\d+\b[^\.\?!]*$|###?\s+|\d+(\.\d+)*\s+[A-Z])', line, re.IGNORECASE)) and not line.endswith(('.', ':', ';', ','))) or
            (len(line) < 45 and not line.endswith(('.', ':', ';', ',')) and line[0].isupper() and not line.startswith('(') and sum(1 for c in line if c.isupper()) >= 2)
        )
        if is_heading:
            if curr_lines:
                sections.append((curr_title, "\n".join(curr_lines)))
                curr_lines = []
            curr_title = re.sub(r'^[#\s]+', '', line).strip()
        else:
            curr_lines.append(line)
    if curr_lines:
        sections.append((curr_title, "\n".join(curr_lines)))
    return sections

def retrieve_relevant_context(query, knowledge, chat_history=None):
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
            
    # Disambiguation: If user specifically asks for 'ubit', avoid pulling general Karachi University info
    if 'ubit' in matched_keys and 'karachi' in matched_keys:
        matched_keys.remove('karachi')
            
    # Context Continuity: If the user didn't mention a university in this turn (e.g. "fees and test strucure"),
    # look back at recent conversation history to identify the university being discussed.
    if not matched_keys:
        # Check LangChain chat_history first
        recent_texts = []
        if chat_history and hasattr(chat_history, 'messages') and chat_history.messages:
            recent_texts.extend([m.content for m in reversed(chat_history.messages)])
        # Check Streamlit session_state messages if available
        if 'messages' in st.session_state and st.session_state.messages:
            # exclude the very last one which is current user prompt
            recent_texts.extend([m.get('content', '') for m in reversed(st.session_state.messages[:-1])])
            
        for text in recent_texts:
            t_lower = text.lower()
            for keyword, key in mapping.items():
                if re.search(rf'\b{re.escape(keyword)}\b', t_lower):
                    matched_keys.add(key)
            if 'ubit' in matched_keys and 'karachi' in matched_keys:
                matched_keys.remove('karachi')
            if matched_keys:
                break
                
    if not matched_keys:
        return "NO_UNI_DETECTED"
    
    stop_words = {'what', 'is', 'the', 'a', 'an', 'for', 'in', 'of', 'to', 'and', 'how', 'are', 'you', 'university', 'tell', 'me', 'about', 'can', 'please', 'karachi', 'details', 'info', 'information'}
    raw_words = re.findall(r'\w+', query_lower)
    # Remove university names themselves from content scoring so query intent words stand out
    uni_name_tokens = {'ned', 'neduet', 'fast', 'nu', 'nuces', 'dow', 'duhs', 'jsmu', 'kmu', 'kmdc', 'smbbmc', 'ku', 'uok', 'sir', 'syed', 'ssuet', 'dawood', 'duet', 'ubit', 'iba', 'aku', 'aga', 'khan', 'agakhan'}
    query_words = [w for w in raw_words if w not in stop_words and w not in uni_name_tokens]
    
    selected_contexts = []
    for uni_key in matched_keys:
        uni_text = knowledge.get(uni_key, "")
        if not uni_text:
            continue
            
        sections = extract_sections(uni_text)
        
        # If user just sent the university name or general greeting (e.g. "AKU", "Tell me about IBA"),
        # provide the top structural sections (overview, programs, and admission/eligibility).
        if not query_words:
            overview_sec = next((f"### {t}\n{c}" for t, c in sections if "overview" in t.lower() or "1." in t.lower()), "")
            prog_sec = next((f"### {t}\n{c}" for t, c in sections if "program" in t.lower() or "undergraduate" in t.lower()), "")
            elig_sec = next((f"### {t}\n{c}" for t, c in sections if "eligib" in t.lower() or "admission" in t.lower()), "")
            best = [s for s in [overview_sec, prog_sec, elig_sec] if s]
            if not best and sections:
                best = [f"### {sections[0][0]}\n{sections[0][1]}"]
        else:
            scored_sections = []
            for title, content in sections:
                comb = (title + " " + content).lower()
                score = 0
                for w in query_words:
                    # Root stem (e.g. fees -> fee, requirements -> requirement)
                    stem = w[:-1] if w.endswith('s') and len(w) > 3 else w
                    if re.search(rf'\b{re.escape(stem)}', title.lower()):
                        score += 25
                    cnt = len(re.findall(rf'\b{re.escape(stem)}', comb))
                    score += min(cnt, 5) * 4
                scored_sections.append((score, title, content))
                
            scored_sections.sort(key=lambda x: x[0], reverse=True)
            # Retrieve up to top 5 sections with positive relevance score
            best = [f"### {t}\n{c}" for s, t, c in scored_sections[:5] if s > 0]
            if not best and sections:
                best = [f"### {sections[0][0]}\n{sections[0][1]}"]
            
        uni_content = "\n\n---\n\n".join(best)
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
            "1. Answer clearly, accurately, and thoroughly using the facts, fees, test patterns, and criteria provided in the Context below.\n"
            "2. If the user mentions one or more universities without a specific question (e.g. 'AKU', 'IBA', 'UBIT', or 'IBA AKU UBIT', 'do you have information of these universities'), confirm that you have full admission information for them, give a brief 1-2 sentence introduction for each from the context, and invite the user to ask about programs, eligibility, entry tests, or fee structures.\n"
            "3. Look carefully across all sections in the Context before claiming information is missing. Only state that you do not have specific information if the provided Context genuinely lacks any mention of it.\n"
            "4. If the context says 'NO_UNI_DETECTED', greet the user warmly and ask which university they would like information about.\n"
            "5. Never hallucinate or mix up details between different universities.\n"
            "6. If asked about your identity, ALWAYS state you are the 'UniChalo AI Assistant'. Do NOT mention Groq, Qwen, Alibaba, or underlying models.\n\n"
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
    # Pass chat history so conversational follow-ups (e.g. "fees and admission") know which university was being discussed
    context = retrieve_relevant_context(user_input, knowledge_db, st.session_state.chat_history)
    
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
with st.sidebar:
    st.header("UniChalo Settings")
    st.write("Supported Universities: NED, FAST, DUHS/Dow, Karachi University, Dawood, SSUET, JSMU, KMU, SMBBMC, UBIT, IBA, AKU.")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chat_history = ChatMessageHistory()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

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
