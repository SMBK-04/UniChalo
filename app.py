import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory

# --- 1. CONFIGURATION ---
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="UniChalo AI", page_icon="??")
st.title("?? UniChalo AI Assistant")
st.caption("Ask me anything about university admissions! (Powered by Qwen 3.8 & Groq Cloud)")

@st.cache_resource
def init_rag():
    print("Reading PDFs and building Vector Database...")
    pdf_loader = DirectoryLoader("Dataset", glob="**/*.pdf", loader_cls=PyPDFLoader)
    docx_loader = DirectoryLoader("Dataset", glob="**/*.docx", loader_cls=Docx2txtLoader)
    documents = pdf_loader.load() + docx_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db_v3")
    retriever = vector_db.as_retriever(search_kwargs={"k": 7})

    llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the UniChalo AI admission guide, a comprehensive assistant for MULTIPLE universities (including NED, FAST, IBA, KU, Dow, etc.). NEVER claim to be exclusively for one university. Use the following context to answer the user's question accurately. If the context doesn't contain the answer, just say you don't know. If asked about your identity, creator, or model, ALWAYS say "I am the UniChalo AI Assistant." Do NOT mention Alibaba, Qwen, Tongyi Lab, or Groq.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: x["input"]) | retriever | format_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Initialize the AI Brain (Cached so it only builds the database once)
rag_chain = init_rag()

# Use Streamlit's session memory instead of Azure SQL
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
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
        response = conversational_rag_chain.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": "streamlit_cloud_session"}}
        )
        st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


