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
regolo.default_model = "Llama-3.3-70B-Instruct"

# testing with mock methods
def test_completions():
    load_dotenv()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_model = "Llama-3.3-70B-Instruct"
    mock_response = "Rome is the capital city of Italy."
    client = regolo.RegoloClient()
    client.static_completions = MagicMock(return_value=mock_response)
    response = client.completions(prompt="Tell me something about Rome.")
    assert response == mock_response


def test_chat_completions():
    load_dotenv()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_model = "Llama-3.3-70B-Instruct"
    mock_response = "Rome is the capital city of Italy."
    client = regolo.RegoloClient()
    client.static_chat_completions = MagicMock(return_value=mock_response)
    response = client.static_chat_completions(prompt="Tell me something about Rome.")
    assert response == mock_response


# testing evaluation of actual response from a client


def test_static_completions() -> None:
    load_dotenv()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_model = "Llama-3.3-70B-Instruct"
    client = regolo.RegoloClient()
    response = client.completions(prompt="Tell me something about Rome.")
    assert type(response) == str


def test_static_chat_completions() -> None:
    load_dotenv()
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_model = "Llama-3.3-70B-Instruct"
    client = regolo.RegoloClient()
    response = client.static_chat_completions(messages=[{"role": "user", "content": "Tell me something about rome"}])
    assert type(response) == tuple

def test_static_image_create() -> None:
    from io import BytesIO
    from PIL import Image

    import regolo

    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_image_model = "FLUX.1-dev"
    client = regolo.RegoloClient()
    img_bytes = client.create_image(prompt="A cat in Rome")[0]
    image = Image.open(BytesIO(img_bytes))
    assert isinstance(image, Image.Image)

def test_static_embeddings() -> None:
    import regolo
    regolo.default_key = os.getenv("TEST_KEY")
    regolo.default_embedder_model ="gte-Qwen2"
    client = regolo.RegoloClient()
    embeddings = client.embeddings(input_text=["test", "test1"])
    assert type(embeddings) == list
