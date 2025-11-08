import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st

import tempfile
from typing import List, Tuple, Dict
from tqdm import tqdm

# Installing Necessary Libraries
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

#Initializing the API Keys
groq_api_key = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

#Set the page and title of the page for streamlit
st.set_page_config(page_title="Bajaj AMC Chatbot (Groq RAG)", layout="wide")
st.title("Bajaj AMC Factsheet Chatbot")

# PDF -→ chunks -→ embeddings -→ VectorDB(FAISS)
def build_or_load_faiss(pdf_file, rebuild=False):
    """
    Extracts text from PDF, chunks it, embeds it with Hugging Face,
    and builds a FAISS vector store.
    """
    index_dir = "faiss_index"
    os.makedirs(index_dir, exist_ok=True)

    # Save uploaded PDF to temporary file
    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_pdf.write(pdf_file.read())
    tmp_pdf.flush()

    # Loading/extracting text from uploaded PDF.
    loader = PyPDFLoader(tmp_pdf.name)
    docs = loader.load()

    # Performing the chunking(Specifically used RecursiveCharacterTextSplitter)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Initialize the embedding (HuggingFace)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Initialize the vector database (FAISS)
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(index_dir)
    return vectordb

# Initialize the LLM (used Groq LLM Model)
llm = ChatGroq(
    model="qwen/qwen3-32b",
    groq_api_key=groq_api_key,
    temperature=0.0
)

system_prompt = (
    "You are an assistant that MUST answer using ONLY the provided context snippets. "
    "If the user's question cannot be answered from the snippets, reply exactly: "
    "\"I don't know — this is not in the factsheet.\". "
    "Do not hallucinate. Always include a short 'Citations:' line listing page numbers "
    "of the snippets used."
)

# It creates a chat messages for an llm. Combines the retrieved context and user question send as an input to LLM
def _messages_(context: List[str], user_question: str) -> List:
    context_text = "\n\n".join(context)[:7000]
    user_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question: {user_question}\n\n"
        "Answer concisely and end with a 'Citations:' line."
    )
    return [SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)]
#============================================================================================================================================================================================================
# Chat Agent class
# This class handles the retrieval and answer generation
class ChatAgent:
    def __init__(self, vectordb: FAISS, llm: ChatGroq, top_k: int = 4):
        self.vectordb = vectordb
        self.llm = llm
        self.top_k = top_k

    def top_k_snippets(self, query:str, k:int = None):
        k = k or self.top_k
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)
        formatted = []
        for d, score in docs_and_scores:
            page = d.metadata.get("page", "?")
            snippet = d.page_content.strip().replace("\n", " ")
            formatted.append((f"Page {page}: {snippet}", score, d))
        return formatted

    def answer(self, query: str) -> Dict:
        retrieved = self.top_k_snippets(query)
        context = [t[0] for t in retrieved]
        messages = _messages_(context, query)
        try:
            result = self.llm.invoke(messages)  #It will generate the answer using recieved context
            text = result.content
        except Exception as e:
            text = f"Error calling Groq API: {e}"
        sources = [
            {"page": doc.metadata.get("page"),
             "score": float(score),
             "snippet": s[:400]}
            for s, score, doc in retrieved
        ]
        return {"answer": text, "sources": sources}

# Streamlit UI
pdf_file = st.sidebar.file_uploader("Please Upload PDF file", type=["pdf"])
if pdf_file:
    with st.spinner("Extracting text from PDF and building FAISS index..."):
        vectordb = build_or_load_faiss(pdf_file)
    st.success("Document indexed successfully!")
else:
    st.warning("Please upload the PDF file to continue.")
    st.stop()

agent = ChatAgent(vectordb=vectordb, llm=llm)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input("How may i help you?....")
if user_query:
    st.session_state.chat_history.append(("user", user_query))
    with st.spinner("Generating answer…"):
        resp = agent.answer(user_query)
    st.session_state.chat_history.append(("AI", resp["answer"]))
    st.session_state.chat_history.append(("source", resp["sources"]))

# Display messages
for role, msg in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.write(msg)
    elif role == "AI":
        with st.chat_message("assistant"):
            st.write(msg)
    elif role == "source":
        with st.expander("Sources"):
            for s in msg:
                st.write(f"page {s['page']} (score {round(s['score'],3)}): {s['snippet']}")

st.markdown("---")
st.caption("Built for Bajaj AMC Factsheet QA")
