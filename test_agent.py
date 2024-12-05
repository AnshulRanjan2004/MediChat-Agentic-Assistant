import agent 

def test_router():
    from unittest.mock import Mock

    mock_llm = Mock()
    mock_llm.side_effect = lambda prompt: {
        "recommend": "recommend",
        "qa": "qa",
        "summary": "summary",
        "alternative": "alternative",
    }[prompt.split("Query:")[1].strip().lower()]

    nodes = {
        "recommend": lambda q: "recommend_node_called",
        "qa": lambda q: "qa_node_called",
        "summary": lambda q: "summary_node_called",
        "alternative": lambda q: "alternative_node_called",
    }

    assert agent.router_node("Can you recommend something?", nodes, mock_llm) == "recommend_node_called"
    assert agent.router_node("What is Ibuprofen?", nodes, mock_llm) == "qa_node_called"
    assert agent.router_node("Summarize Aspirin.", nodes, mock_llm) == "summary_node_called"
    assert agent.router_node("Tell me the weather.", nodes, mock_llm) == "alternative_node_called"
    print("[INFO] All tests passed.")

test_router()