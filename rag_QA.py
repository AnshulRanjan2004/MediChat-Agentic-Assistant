import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import Field
import requests
from typing import Dict, Any


# Custom LLM integration with LMStudio
class LMStudioLLM(LLM):
    endpoint: str = Field(...)  # Declare endpoint as a required field

    def __init__(self, endpoint: str):
        super().__init__(endpoint=endpoint)

    def _call(self, prompt: str, stop: None = None, **kwargs: Any) -> str:
        response = requests.post(
            f"{self.endpoint}/v1/chat/completions",
            json={
                "model": "llama-3.2-3b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
            },
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"LLM Error: {response.status_code} - {response.text}")

    @property
    def _llm_type(self) -> str:
        return "lmstudio"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"endpoint": self.endpoint}


# Test RAG Pipeline
def test_rag_pipeline(qa_chain, query):
    print(f"\n[QUERY] {query}")
    response = qa_chain.invoke({"query": query})
    result = response["result"]
    source_docs = response["source_documents"]

    print(f"[RESPONSE] {result}")
    print("[SOURCES]")
    for doc in source_docs:
        print(f"  - Section: {doc.metadata['section']}, File: {doc.metadata['file']}")


if __name__ == "__main__":
    # Configuration
    persist_directory = "./chroma_db"
    lmstudio_endpoint = "http://127.0.0.1:1234"

    # Load Vector Store
    print("[INFO] Loading vector store...")
    embeddings = OpenAIEmbeddings()  # Use the same embedding function as during vector store creation
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("[INFO] Vector store loaded successfully.")

    # Initialize LLM and RetrievalQA
    print("[INFO] Initializing LLM and RAG pipeline...")
    llm = LMStudioLLM(endpoint=lmstudio_endpoint)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, return_source_documents=True
    )
    print("[INFO] RAG pipeline initialized successfully.")

    # Test Queries
    sample_queries = [
        "What is the composition and primary use of Acetominophen?",
        "Can you list the side effects of Ibuprofen?",
        "What are the contraindications for Aspirin?",
        "How should Metformin be administered?",
        "What are the inactive ingredients in Travoprost Ophthalmic Solution?",
    ]

    for query in sample_queries:
        test_rag_pipeline(qa_chain, query)
