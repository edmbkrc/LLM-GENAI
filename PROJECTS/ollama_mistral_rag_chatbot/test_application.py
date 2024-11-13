from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""

def query_and_validate(question: str, expected_response: str):
    response_text = query_rag(question)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )
    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt).strip().lower()
    return "true" in evaluation_results_str

def test_monopoly_rules():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly?",
        expected_response="$1500"
    )

def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride?",
        expected_response="10 points"
    )
