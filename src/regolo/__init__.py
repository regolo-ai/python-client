# The version of the current module.
__version__ = "1.9.1"

import os

# Default values for the API key and model.
# These can be set later before creating instances of the client.
default_key = None
default_chat_model = None
default_image_generation_model = None
default_embedder_model = None
default_audio_transcription_model = None
default_reranker_model = None

# if your server has a /models openai compatible endpoint, you can enable model availability checks
enable_model_checks = True

# Importing the main client from the regolo.client module for interacting with the Regolo API.
from regolo.client.regolo_client import RegoloClient

# Exposing specific methods from the RegoloClient class to make them easily accessible.
# These methods are used for interacting with the completions and chat completions APIs.

# Static method for getting completions (text-based response from the model).
static_completions = RegoloClient.static_completions

# Static method for getting chat-based completions (message-based response from the model).
static_chat_completions = RegoloClient.static_chat_completions

# static method for creating an image

static_image_create = RegoloClient.static_create_image

# static method for generating embeddings

static_embeddings = RegoloClient.static_embeddings

# static method for generating an audio transcription

static_audio_transcription = RegoloClient.static_audio_transcription

def key_load_from_env_if_exists():
    """
    Method, which will update default_key if its environment variable is set.
    """
    global default_key
    default_key = default_key if os.getenv("API_KEY") is None else os.getenv("API_KEY")

def default_chat_model_load_from_env_if_exists():
    """
    Method, which will update default_model if its environment variable is set.
    """
    global default_chat_model
    default_chat_model = default_chat_model if os.getenv("LLM") is None else os.getenv("LLM")

def default_image_model_load_from_env_if_exists():
    """
    Method, which will update default_image_model if its environment variable is set.
    """
    global default_image_generation_model
    default_image_generation_model = default_image_generation_model if os.getenv("IMAGE_GENERATION_MODEL") is None \
        else os.getenv("IMAGE_GENERATION_MODEL")

def default_embedder_model_load_from_env_if_exists():
    """
    Method, which will update default_embedder_model if its environment variable is set.
    """
    global default_embedder_model
    default_embedder_model = default_embedder_model if os.getenv("EMBEDDER_MODEL") is None \
        else os.getenv("EMBEDDER_MODEL")

def default_audio_transcription_model_load_from_env_if_exists():
    """
    Method, which will update default_embedder_model if its environment variable is set.
    """
    global default_audio_transcription_model
    default_audio_transcription_model = default_audio_transcription_model if os.getenv("AUDIO_TRANSCRIPTION_MODEL") is None \
        else os.getenv("AUDIO_TRANSCRIPTION_MODEL")


def try_loading_from_env():
    """
    Method, which will update default values from environment variables if the environment variables are set.
    For each default value, if the corresponding envs aren't set, it will leave them as they are.
    """
    key_load_from_env_if_exists()
    default_chat_model_load_from_env_if_exists()
    default_image_model_load_from_env_if_exists()
    default_embedder_model_load_from_env_if_exists()
    default_audio_transcription_model_load_from_env_if_exists()
