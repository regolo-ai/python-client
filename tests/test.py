import os
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv

import regolo

load_dotenv()

API_KEY = os.environ.get("TEST_KEY")

pytest.mark.skipif(API_KEY is None, reason="Api key not set")

# Set default key and model for testing
regolo.default_key = os.getenv("TEST_KEY")
regolo.default_chat_model = "Llama-3.1-8B-Instruct"

# testing with mock methods
def test_completions():
    load_dotenv()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_chat_model = "Llama-3.1-8B-Instruct"
    mock_response = "Rome is the capital city of Italy."
    client = regolo.RegoloClient()
    client.static_completions = MagicMock(return_value=mock_response)
    response = client.completions(prompt="Tell me something about Rome.")
    assert response == mock_response


def test_chat_completions():
    load_dotenv()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_chat_model = "Llama-3.1-8B-Instruct"
    mock_response = "Rome is the capital city of Italy."
    client = regolo.RegoloClient()
    client.static_chat_completions = MagicMock(return_value=mock_response)
    response = client.static_chat_completions(prompt="Tell me something about Rome.")
    assert response == mock_response


# testing evaluation of actual response from a client


def test_static_completions() -> None:
    load_dotenv()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_chat_model = "Llama-3.1-8B-Instruct"
    client = regolo.RegoloClient()
    response = client.completions(prompt="Tell me something about Rome.")
    assert type(response) == str


def test_static_chat_completions() -> None:
    load_dotenv()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_model = "Llama-3.1-8B-Instruct"
    client = regolo.RegoloClient()
    response = client.static_chat_completions(messages=[{"role": "user", "content": "Tell me something about rome"}])
    assert type(response) == tuple

def test_static_image_create() -> None:
    from io import BytesIO

    from PIL import Image

    import regolo

    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_image_generation_model = "Qwen-Image"
    client = regolo.RegoloClient()
    img_bytes = client.create_image(prompt="A cat in Rome")[0]
    image = Image.open(BytesIO(img_bytes))
    assert isinstance(image, Image.Image)

def test_static_embeddings() -> None:
    import regolo
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_embedder_model ="Qwen3-Embedding-8B"
    client = regolo.RegoloClient()
    embeddings = client.embeddings(input_text=["test", "test1"])
    assert type(embeddings) == list


def test_rerank():
    load_dotenv()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_reranker_model = "jina-reranker-v2"
    mock_response = [
        {"index": 0, "relevance_score": 0.95, "document": "Rome is the capital of Italy"},
        {"index": 1, "relevance_score": 0.72, "document": "Paris is the capital of France"}
    ]
    client = regolo.RegoloClient()
    client.static_rerank = MagicMock(return_value=mock_response)
    documents = ["Rome is the capital of Italy", "Paris is the capital of France"]
    response = client.rerank(query="What is the capital of Italy?", documents=documents)
    assert response == mock_response


"""
def test_audio_transcriptions() -> None:
    import regolo
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_audio_transcription_model = "faster-whisper-large-v3"
    client = regolo.RegoloClient()
    transcriptions =  client.audio_transcription(file="<example_file_path>")
    assert type(transcriptions) == dict
"""

test_completions()
test_chat_completions()
test_static_completions()
test_static_chat_completions()
test_static_image_create()
test_static_embeddings()
test_rerank()
