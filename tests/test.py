import os
from unittest.mock import MagicMock
import regolo
import pytest
import dotenv


dotenv.load()

API_KEY = os.environ.get("TEST_KEY")

pytest.mark.skipif(API_KEY is None, reason="Api key not set")

# Set default key and model for testing
regolo.default_key = os.getenv("TEST_KEY")
regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"

def test_completions():
    dotenv.load()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"
    mock_response = "Rome is the capital city of Italy."
    client = regolo.RegoloClient()
    client.static_completions = MagicMock(return_value=mock_response)
    response = client.completions(prompt="Tell me something about Rome.")
    assert response == mock_response


def test_chat_completions():
    dotenv.load()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"
    mock_response = "Rome is the capital city of Italy."
    client = regolo.RegoloClient()
    client.static_chat_completions = MagicMock(return_value=mock_response)
    response = client.static_chat_completions(prompt="Tell me something about Rome.")
    assert response == mock_response


def test_static_completions():
    dotenv.load()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"
    client = regolo.RegoloClient()
    response = client.completions(prompt="Tell me something about Rome.")
    assert type(response) == str


def test_static_chat_completions():
    dotenv.load()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"
    client = regolo.RegoloClient()
    response = client.static_chat_completions(messages=[{"role": "user", "content": "Tell me something about rome"}])
    assert type(response) == tuple