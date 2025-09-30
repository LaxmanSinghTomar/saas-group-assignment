from unittest.mock import patch

from evals.evals_generation import GenerationSettings, generate_response
from evals.evals_retrieval import hybrid_search


@patch("evals.evals_generation.OpenAI")
def test_generation_uses_context(mock_client, hybrid_index):
    mock_client.return_value.chat.completions.create.return_value.choices = [
        type("obj", (), {"message": type("obj", (), {"content": "To change your password, go to account settings."})})
    ]
    response = hybrid_search("How do I change my password?", hybrid_index, top_k=1)
    generation = generate_response(response, settings=GenerationSettings(temperature=0.0))
    assert "password" in generation["answer"].lower()
    assert generation["metadata"]["context_used"] == 1


