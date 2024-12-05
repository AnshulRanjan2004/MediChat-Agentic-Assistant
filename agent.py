import os
from typing import List, Dict, Any
from langchain.llms.base import LLM
from langchain.agents import Tool
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from pydantic import Field
import requests
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="Intelligent Agent Assistant",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stApp {
        background-color: #2b2b2b;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
        color: white;
    }
    .sidebar .sidebar-content a {
        color: #1e90ff;
        text-decoration: none;
    }
    .chat-bubble-user {
        text-align: right;
        background-color: #4caf50;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 70%;
        display: inline-block;
    }
    .chat-bubble-assistant {
        text-align: left;
        background-color: #3a3a3a;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        max-width: 70%;
        display: inline-block;
    }
    .stTextInput label {
        color: white;
    }
    .stButton button {
        background-color: #1e90ff;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #1c86ee;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with logo and information
with st.sidebar:
    st.image("image.png", use_column_width=True)  # Replace with your logo URL
    st.title("🤖 Intelligent Agent Assistant")
    st.markdown("""
    **Welcome to your AI-powered assistant!**
    
    - **Summarizer**: Get concise summaries.
    - **Recommender**: Receive personalized recommendations.
    - **QA**: Ask factual questions.
    - **Alternative Search**: Search the web for information.
    """)
    st.markdown("---")
    st.markdown("Developed by Team 5.")

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# Custom LLM integration with LMStudio
class LMStudioLLM(LLM):
    endpoint: str = Field(...)  # Declare endpoint as a required field

    def __init__(self, endpoint: str):
        super().__init__(endpoint=endpoint)

    def _call(self, prompt: str, stop: None = None, **kwargs: Any) -> str:
        try:
            import time
            start_time = time.time()
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": "llama-3.2-3b-instruct",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                },
                timeout=100  # Add a timeout
            )
            end_time = time.time()
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"LLM Error: {response.status_code} - {response.text}")
        except requests.exceptions.Timeout:
            raise Exception("[ERROR] LLM request timed out.")
        except Exception as e:
            raise Exception(f"[ERROR] An unexpected error occurred: {str(e)}")

    @property
    def _llm_type(self) -> str:
        return "lmstudio"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"endpoint": self.endpoint}

# Streamlit Caching for resources
@st.cache_resource
def load_vector_store(persist_directory: str):
    st.write("[INFO] Loading vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    st.write("[INFO] Vector store loaded successfully.")
    return vector_store

@st.cache_resource
def initialize_llm(endpoint: str):
    st.write("[INFO] Initializing LLM...")
    llm = LMStudioLLM(endpoint=endpoint)
    st.write("[INFO] LLM initialized successfully.")
    return llm

# Tool Wrappers
def summarize(query: str) -> str:
    from summarizer import optimized_summarizer

    try:
        vector_store = load_vector_store(persist_directory="./chroma_db")
        llm = initialize_llm(endpoint="http://127.0.0.1:1234")

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        st.write(f"[INFO] Processing query: {query}")
        result = optimized_summarizer(query, retriever, llm)
        return f"[Tool: Summarizer] {result}"

    except Exception as e:
        return f"[Tool: Summarizer] An error occurred: {str(e)}"

def recommend(query: str) -> str:
    from recommend import rag_recommender

    try:
        vector_store = load_vector_store(persist_directory="./chroma_db")
        llm = initialize_llm(endpoint="http://127.0.0.1:1234")

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        st.write(f"[INFO] Processing query: {query}")
        result = rag_recommender(query, retriever, llm)
        return f"[Tool: Recommender] {result}"

    except Exception as e:
        return f"[Tool: Recommender] An error occurred: {str(e)}"

def alternative(query: str) -> str:
    from alternative import initialize_web_search_agent

    try:
        llm = initialize_llm(endpoint="http://127.0.0.1:1234")

        st.write("[INFO] Initializing Web Search Agent...")
        web_search_agent = initialize_web_search_agent(llm)
        st.write("[INFO] Web Search Agent initialized successfully.")

        st.write(f"[INFO] Processing query: {query}")
        result = web_search_agent.run(query)
        return f"[Tool: Alternative Search] {result}"

    except Exception as e:
        return f"[Tool: Alternative Search] An error occurred: {str(e)}"

def qa(query: str) -> str:
    from rag_QA import test_rag_pipeline

    try:
        vector_store = load_vector_store(persist_directory="./chroma_db")
        llm = initialize_llm(endpoint="http://127.0.0.1:1234")

        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, return_source_documents=True
        )

        st.write(f"[INFO] Processing query: {query}")
        result = test_rag_pipeline(qa_chain, query)

        if "I don't know" in result or "No relevant information found" in result:
            return None  # Indicate failure to find an answer
        else:
            return f"[Tool: QA] {result}"

    except Exception as e:
        return f"[Tool: QA] An error occurred: {str(e)}"

def classify_query(query: str, llm: LLM) -> str:
    if "recommend" in query.lower() or "recommendation" in query.lower():
        return "Recommender"

    question_words = ["what", "how", "when", "where", "why", "who", "can", "is", "are", "do", "does", "did", "list", "which", "whom", "whose"]
    if any(query.lower().startswith(word) for word in question_words):
        return "QA"

    prompt = f"""
You are an intelligent query classifier for an agent application. The application has four tools:
1. Summarizer: For queries seeking a concise summary of information, even indirectly.
2. Recommender: For queries requesting recommendations or alternatives.
3. QA: For factual questions requiring precise answers.
4. Alternative Search: For all other types of queries.

Based on the above tools, classify the following query:
Query: "{query}"

Respond only with one of these tool names: Summarizer, Recommender, QA, or Alternative Search.
    """
    response = llm(prompt).strip()
    valid_tools = {"Summarizer", "Recommender", "QA", "Alternative Search"}
    return response if response in valid_tools else "QA"  # Default to QA

# Main App
def main():
    st.markdown("<h1 style='text-align: center; color: white;'>🤖 Intelligent Agent Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: white;'>Your AI-powered assistant for information and recommendations.</h4>", unsafe_allow_html=True)
    st.markdown("---")

    query = st.text_input("Ask me anything...", key="input")

    if st.button("Send") and query.strip() != "":
        st.session_state['messages'].append({"role": "user", "content": query})

        try:
            llm = initialize_llm(endpoint="http://127.0.0.1:1234")
            tool_name = classify_query(query, llm)
            st.write(f"[INFO] Routed to tool: {tool_name}")

            tools = {
                "Summarizer": summarize,
                "Recommender": recommend,
                "QA": qa,
                "Alternative Search": alternative,
            }

            if tool_name in tools:
                response = tools[tool_name](query)

                if tool_name == "QA" and (response is None or "An error occurred" in response):
                    st.write("[INFO] QA tool could not find an answer. Switching to Alternative Search.")
                    response = tools["Alternative Search"](query)

                st.session_state['messages'].append({"role": "assistant", "content": response})
            else:
                st.write("[ERROR] No matching tool found for the query.")
        except Exception as e:
            st.session_state['messages'].append({"role": "assistant", "content": f"An error occurred: {str(e)}"})

    for message in st.session_state['messages']:
        if message['role'] == 'user':
            st.markdown(f"<div class='chat-bubble-user'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-assistant'>{message['content']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
