import os
import requests
from typing import Dict, Any, List
from langchain.llms.base import LLM
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic import Field

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
            print(f"[DEBUG] LLM call duration: {end_time - start_time:.2f} seconds")
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

# Optimized Summarizer Function
def optimized_summarizer(query: str, retriever, llm: LLM) -> str:
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)

    if not docs:
        return "No relevant documents found to summarize."

    # Combine documents into a single string
    combined_docs = "\n\n".join([doc.page_content for doc in docs])

    # Prepare the summarization prompt
    prompt = f"""You are an expert medical summarizer. Read the following documents and provide a concise and informative summary relevant to the query "{query}".

Documents:
{combined_docs}

Summary:"""

    # Call the LLM once to get the summary
    summary = llm(prompt)

    return summary.strip()

if __name__ == "__main__":
    # Configuration
    persist_directory = "./chroma_db"
    lmstudio_endpoint = "http://127.0.0.1:1234"

    # Load Vector Store
    print("[INFO] Loading vector store...")
    embeddings = OpenAIEmbeddings()  # Use the same embedding function as during vector store creation
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    print("[INFO] Vector store loaded successfully.")

    # Initialize LLM
    print("[INFO] Initializing LLM...")
    llm = LMStudioLLM(endpoint=lmstudio_endpoint)
    print("[INFO] LLM initialized successfully.")

    # Initialize retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Reduced k to 3

    # Test Queries
    test_queries = [
        "Summarize the details of Amoxicillin."
    ]

    for query in test_queries:
        print(f"\n[QUERY]: {query}")
        summary = optimized_summarizer(query, retriever, llm)
        print(f"[SUMMARY]: {summary}")
