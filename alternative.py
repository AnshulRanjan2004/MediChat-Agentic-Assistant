import os
import requests
from typing import Dict, Any, List
from langchain.llms.base import LLM
from langchain_community.tools import DuckDuckGoSearchResults, Tool
from langchain.agents import initialize_agent
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

# Initialize the DuckDuckGo Search Tool
ddg_search = DuckDuckGoSearchResults()

tools = [
    Tool(
        name="DuckDuckGo Search",
        func=ddg_search.run,
        description="Use this tool to browse information from the Internet.",
    )
]

# Initialize the Web Search Summarizer Agent
def initialize_web_search_agent(llm: LLM) -> Any:
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=True,
    )

# Main Function
if __name__ == "__main__":
    # LLM Configuration
    lmstudio_endpoint = "http://127.0.0.1:1234"

    # Initialize LLM
    print("[INFO] Initializing LLM...")
    llm = LMStudioLLM(endpoint=lmstudio_endpoint)
    print("[INFO] LLM initialized successfully.")

    # Initialize Web Search Agent
    print("[INFO] Initializing Web Search Agent...")
    web_search_agent = initialize_web_search_agent(llm)
    print("[INFO] Web Search Agent initialized successfully.")

    # Example Queries
    test_queries = [
        "Weather in London in the coming 3 days",
        "Latest advancements in AI technology",
        "Symptoms and treatments for seasonal allergies",
        "How to build a Kubernetes cluster on AWS",
        "Top tourist attractions in Paris",
        "What is the capital of Australia?",
        "Best practices for Python programming",
        "Overview of COVID-19 vaccines",
        "What are the benefits of yoga?",
        "Upcoming space missions by NASA",
    ]

    # Run Queries
    for query in test_queries:
        print(f"\n[QUERY]: {query}")
        try:
            response = web_search_agent.run(query)
            print(f"[RESPONSE]: {response}")
        except Exception as e:
            print(f"[ERROR]: {e}")
