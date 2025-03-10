# The version of the current module.
__version__ = "1.0.3"

# Default values for the API key and model.
# These can be set later before creating instances of the client.
default_key = None
default_model = None
default_image_model = None
default_embedder_model = None

# if your server has a /models openai compatible endpoint, you can enable model availability checks
enable_model_checks = False

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
