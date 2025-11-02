import json
import os
from base64 import b64decode
from types import GeneratorType
from typing import Any, Callable, Dict, Generator, List, Optional, TypeAlias

import httpx
from json_repair import json_repair

import regolo
from regolo.instance.regolo_instance import RegoloInstance
from regolo.instance.structures.conversation_model import Conversation, ConversationLine
from regolo.keys.keys import KeysHandler
from regolo.models.models import ModelsHandler

REGOLO_URL = "https://api.regolo.ai"
COMPLETIONS_URL_PATH = "/v1/completions"
CHAT_COMPLETIONS_URL_PATH = "/v1/chat/completions"
IMAGE_GENERATION_URL_PATH = "/v1/images/generations"
EMBEDDINGS_URL_PATH = "/v1/embeddings"
AUDIO_TRANSCRIPTION_URL_PATH = "/v1/audio/transcriptions"
RERANK_URL_PATH = "/v1/rerank"

os.environ["REGOLO_URL"] = REGOLO_URL \
    if os.getenv("REGOLO_URL") is None \
    else os.getenv("REGOLO_URL")

os.environ["COMPLETIONS_URL_PATH"] = COMPLETIONS_URL_PATH \
    if os.getenv("COMPLETIONS_URL_PATH") is None \
    else os.getenv("COMPLETIONS_URL_PATH")

os.environ["CHAT_COMPLETIONS_URL_PATH"] = CHAT_COMPLETIONS_URL_PATH \
    if os.getenv("CHAT_COMPLETIONS_URL_PATH") is None \
    else os.getenv("CHAT_COMPLETIONS_URL_PATH")

os.environ["IMAGE_GENERATION_URL_PATH"] = IMAGE_GENERATION_URL_PATH \
    if os.getenv("IMAGE_GENERATION_URL_PATH") is None \
    else os.getenv("IMAGE_GENERATION_URL_PATH")

os.environ["EMBEDDINGS_URL_PATH"] = EMBEDDINGS_URL_PATH \
    if os.getenv("EMBEDDINGS_URL_PATH") is None \
    else os.getenv("EMBEDDINGS_URL_PATH")

os.environ["AUDIO_TRANSCRIPTION_URL_PATH"] = AUDIO_TRANSCRIPTION_URL_PATH \
    if os.getenv("AUDIO_TRANSCRIPTION_URL_PATH") is None \
    else os.getenv("AUDIO_TRANSCRIPTION_URL_PATH")

os.environ["RERANK_URL_PATH"] = RERANK_URL_PATH \
    if os.getenv("RERANK_URL_PATH") is None \
    else os.getenv("RERANK_URL_PATH")


timeout = 500

Role: TypeAlias = str
Content: TypeAlias = str


def safe_post(client: httpx.Client,
              url_to_query: str,
              json_to_query: Optional[dict] = None,
              headers_to_query: Optional[dict] = None) -> httpx.Response:
    """
    Sends a POST request using the provided HTTPX client.

    :param client: The instance of an HTTPX client to use for sending the request.
    :param url_to_query: The URL to which the POST request is sent.
    :param json_to_query: The JSON payload to include in the request body. (Optional)
    :param headers_to_query: The headers to include in the request. (Optional)

    :return: The HTTPX response object.

    :raises httpx.HTTPStatusError: If the request fails with an HTTP error status.
    """

    response = client.post(url=url_to_query, json=json_to_query, headers=headers_to_query, timeout=timeout)
    response.raise_for_status()
    return response


class RegoloClient:
    def __init__(self,
                 chat_model: Optional[str] = None,
                 embedder_model: Optional[str] = None,
                 image_generation_model: Optional[str] = None,
                 audio_transcription_model: Optional[str] = None,
                 reranker_model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 alternative_url: Optional[str] = None,
                 pre_existent_conversation: Optional[Conversation] = None,
                 pre_existent_client: httpx.Client = None) -> None:
        """
        Initialize the client for regolo.ai HTTP API.

        :param chat_model: The regolo.ai chat model to use.
            (Defaults to regolo.default_model)
        :param embedder_model: The regolo.ai embedder model to use for text embeddings.
            (Optional)
        :param image_generation_model: The regolo.ai model to use for image generation.
            (Optional)
        :param audio_transcription_model: The regolo.ai model to use for audio transcription.
            (Optional)
        :param reranker_model: The regolo.ai model to use for reranking search results.
            (Optional)
        :param api_key: The API key for regolo.ai.
            (Defaults to regolo.default_key)
        :param alternative_url: Base URL of the regolo HTTP server.
            (Optional)
        :param pre_existent_conversation: An existing conversation instance to continue chatting with.
            (Optional)
        :param pre_existent_client: An existing httpx.Client instance to use.
            (Optional)
        """

        model = regolo.default_chat_model if chat_model is None else chat_model
        embedder_model = regolo.default_embedder_model if embedder_model is None else embedder_model
        image_generation_model = regolo.default_image_generation_model if image_generation_model is None else image_generation_model
        audio_transcription_model = regolo.default_audio_transcription_model if audio_transcription_model is None else audio_transcription_model
        api_key = regolo.default_key if api_key is None else api_key
        base_url = None if alternative_url is None else alternative_url
        client = httpx.Client(base_url=os.getenv(
            "REGOLO_URL") if base_url is None else base_url) if pre_existent_client is None else pre_existent_client

        self.instance = RegoloInstance(chat_model=model,
                                       embedder_model=embedder_model,
                                       image_generation_model=image_generation_model,
                                       audio_transcription_model=audio_transcription_model,
                                       api_key=api_key,
                                       previous_conversations=pre_existent_conversation,
                                       reranker_model=reranker_model,
                                       client=client,
                                       base_url=base_url)

    @classmethod
    def from_instance(cls, instance: RegoloInstance, alternative_url: Optional[str] = None) -> "RegoloClient":
        """Creates RegoloClient from instance."""
        return cls(api_key=instance.api_key, chat_model=instance.chat_model, alternative_url=alternative_url,
                   pre_existent_client=instance.client, pre_existent_conversation=instance.conversation)

    def change_model(self, model: str) -> None:
        """Change model used in this instance of regolo_client"""
        try:
            self.instance.change_model(new_model=model)
        except Exception as e:
            print(e)

    @staticmethod
    def get_available_models(api_key: str,
                             base_url: str = os.getenv("REGOLO_URL"),
                             model_info: bool=False) -> List[str] | List[dict]:
        """
        Gets all available models on regolo.ai.

        :param base_url: Base URL of the regolo HTTP server.
        :param api_key: The API key for regolo.ai (Defaults to regolo.default_key if it exists)
        :param model_info: Whether to retrieve information about the model (Defaults to False)

        :return model_info=False: A list of available models (list[str])
        :return model_info=True: A list of available models and their information (list[dict])
        """

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        return ModelsHandler.get_models(base_url=base_url, api_key=api_key, model_info=model_info)

    @staticmethod
    def create_stream_generator(client: httpx.Client,
                                base_url: str,
                                payload: Optional[dict] = None,
                                files: Optional[dict] = None,
                                data: Optional[dict] = None,
                                headers: dict = None,
                                full_output: bool = False,
                                search_url: str = "",
                                output_handler: Callable[[Dict], Any] = None) -> Generator[Any, Any, None]:
        """
        Yields generators for streams from regolo.ai (generalized for JSON and multipart requests).

        :param client: The httpx.Client instance to use.
        :param base_url: Base URL of the regolo HTTP server.
        :param payload: The JSON request payload to send (for JSON requests).
            (Optional)
        :param files: The files dict for multipart requests.
            (Optional)
        :param data: The form data dict for multipart requests.
            (Optional)
        :param headers: The request headers.
        :param full_output: Whether to return the full response.
        :param search_url: The URL for the search request.
        :param output_handler: A function that processes responses if full_output=False.
        :return: A generator that yields streamed responses from regolo.ai.
        """
        # Determine the request type and prepare arguments
        request_kwargs = {
            "url": f"{base_url}{search_url}",
            "headers": headers
        }

        if payload is not None:
            # JSON request
            request_kwargs["json"] = payload
        elif files is not None and data is not None:
            # Multipart form data request
            request_kwargs["files"] = files
            request_kwargs["data"] = data
        else:
            raise ValueError("Either payload (for JSON) or both files and data (for multipart) must be provided")

        with client.stream("POST", **request_kwargs) as response:
            if response.status_code != 200:
                raise Exception(f"Error: Received unexpected status code {response.status_code}")

            # Iterate over complete lines instead of raw bytes
            for line in response.iter_lines():
                if not line:
                    continue

                # Decode if necessary and remove the "data:" prefix if present
                decoded_line = line.decode("utf-8") if isinstance(line, bytes) else line
                decoded_line = decoded_line.strip()
                if decoded_line == "data: [DONE]":
                    break
                if decoded_line.startswith("data:"):
                    decoded_line = decoded_line[len("data:"):].strip()

                try:
                    # Repair and parse the JSON chunk
                    data_chunk = json.loads(json_repair.repair_json(decoded_line))
                except (Exception,):
                    continue

                if full_output:
                    yield data_chunk
                else:
                    # Handle both dict and list responses uniformly
                    yield output_handler(data_chunk)

    # Completions

    @staticmethod
    def static_completions(prompt: str,
                           model: Optional[str] = None,
                           api_key: Optional[str] = None,
                           stream: bool = False,
                           max_tokens: int = 200,
                           temperature: Optional[float] = 0.5,
                           top_p: Optional[float] = None,
                           top_k: Optional[int] = None,
                           client: Optional[httpx.Client] = None,
                           base_url: str = os.getenv("REGOLO_URL"),
                           full_output: bool = False) -> str | Generator[Any, Any, None]:
        """
        Will return generators for stream=True and values for stream=False
        Send a prompt to regolo server and get the generated response.

        :param prompt: The input prompt to the LLM.
        :param model: The regolo.ai model to use. (Optional)
        :param api_key: The API key for regolo.ai. (Optional)
        :param stream: Whether to stream the prompt from regolo.ai. (Defaults to False)
        :param max_tokens: Maximum number of tokens to generate. (Defaults to 200)
        :param temperature: Sampling temperature for randomness. (Optional)
        :param top_p: Nucleus sampling parameter. (Optional)
        :param top_k: Top-k sampling parameter. (Optional)
        :param client: httpx client to use. (Optional)
        :param base_url: Base URL of the regolo HTTP server. (Defaults to REGOLO_URL)
        :param full_output: Whether to return the full response. (Defaults to False)

        :return for stream=true, full_output=False: Generator, which yields dicts with the responses from regolo.ai.
        :return for stream=True, full_output=True: Generator, which yields tuples of Role, Content of response.
        :return for stream=False, full_output=False: String with response from regolo.ai.
        :return for stream=False, full_output=True: String containing the text of response.
        """

        def handle_search_text_completions(data: dict) -> str:
            """
            Internal method, describes how RegoloClient.create_stream_generator() should handle output from completions.
            """
            return data.get("choices", [{}])[0].get("text")

        # Use the default API key if none is provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        # Use the default model if none is specified
        if model is None:
            model = regolo.default_chat_model

        # Validate the selected model
        ModelsHandler.check_model(model=model, api_key=api_key, base_url=base_url)

        # Create a new HTTP client if none is provided
        if client is None:
            client = httpx.Client()

        # Construct the payload for the API request
        payload = {
            "prompt": prompt,
            "stream": stream,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        # Remove None values from payload to avoid unnecessary parameters
        payload = {k: v for k, v in payload.items() if v is not None}

        # Set authorization header
        headers = {"Authorization": api_key}

        if stream:
            # If streaming is enabled, return a generator for handling streamed responses
            return RegoloClient.create_stream_generator(
                client=client,
                base_url=base_url,
                payload=payload,
                headers=headers,
                full_output=full_output,
                search_url=os.getenv("COMPLETIONS_URL_PATH"),
                output_handler=handle_search_text_completions
            )
        else:
            # Send a synchronous POST request
            response = safe_post(
                client=client,
                url_to_query=f"{base_url}{os.getenv("COMPLETIONS_URL_PATH")}",
                json_to_query=payload,
                headers_to_query=headers
            )

            if full_output:
                # Return the full JSON response if requested
                return response.json()
            else:
                # Extract and return only the generated text
                return response.json()["choices"][0]["text"]

    def completions(self,
                    prompt: str,
                    stream: bool = False,
                    max_tokens: int = 200,
                    temperature: Optional[float] = None,
                    top_p: Optional[float] = None,
                    top_k: Optional[int] = None,
                    full_output: bool = False) -> str | GeneratorType:
        """
        Will return generators for stream=True and values for stream=False
        Performs requests to completions endpoint from RegoloClient instance.

        :param prompt: The input prompt to the LLM.
        :param stream: Whether to stream the prompt from regolo.ai. (Defaults to False)
        :param max_tokens: Maximum number of tokens to generate. (Defaults to 200)
        :param temperature: Sampling temperature for randomness. (Optional)
        :param top_p: Nucleus sampling parameter. (Optional)
        :param top_k: Top-k sampling parameter. (Optional)
        :param full_output: Whether to return the full response. (Defaults to False)

        :return for stream=true, full_output=False: Generator, which yields dicts with the responses from regolo.ai.
        :return for stream=True, full_output=True: Generator, which yields tuples of Role, Content of response.
        :return for stream=False, full_output=False: String with response from regolo.ai.
        :return for stream=False, full_output=True: String containing the text of response.
        """
        if self.instance.get_base_url() is None:
            base_url = os.getenv("REGOLO_URL")
        else:
            base_url = self.instance.get_base_url()
        response = self.static_completions(prompt=prompt,
                                           model=self.instance.get_model(),
                                           api_key=self.instance.get_api_key(),
                                           stream=stream,
                                           max_tokens=max_tokens,
                                           temperature=temperature,
                                           top_p=top_p,
                                           top_k=top_k,
                                           client=self.instance.get_client(),
                                           base_url=base_url,
                                           full_output=full_output)

        return response

    # Chat completions

    @staticmethod
    def static_chat_completions(messages: Conversation | List[Dict[str, str]],
                                model: Optional[str] = None,
                                api_key: Optional[str] = None,
                                stream: bool = False,
                                max_tokens: int = 200,
                                temperature: Optional[float] = None,
                                top_p: Optional[float] = None,
                                top_k: Optional[int] = None,
                                client: Optional[httpx.Client] = None,
                                base_url: str = os.getenv("REGOLO_URL"),
                                full_output: bool = False
                                ) -> Generator[Any, Any, None] | tuple[Role, Content] | dict:
        """
        Sends a series of chat messages to the vLLM server and gets the response.

        :param messages: A list of messages in the format [{"role": "user"|"assistant", "content": "message"}].
        :param model: The regolo.ai model to use. (Optional)
        :param api_key: The API key for regolo.ai. (Optional)
        :param stream: Whether to stream the prompt from regolo.ai. (Defaults to False)
        :param max_tokens: Maximum number of tokens to generate. (Defaults to 200)
        :param temperature: Sampling temperature for randomness. (Optional)
        :param top_p: Nucleus sampling parameter. (Optional)
        :param top_k: Top-k sampling parameter. (Optional)
        :param client: httpx client to use. (Optional)
        :param base_url: Base URL of the regolo HTTP server. (Defaults to REGOLO_URL)
        :param full_output: Whether to return full response. (Defaults to False)

        :return for stream=true, full_output=False: Generator, which yields dicts with the responses from regolo.ai.
        :return for stream=True, full_output=True: Generator, which yields tuples of Role, Content of response.
        :return for stream=False, full_output=False: String, with response from regolo.ai.
        :return for stream=False, full_output=True: Tuple, which consists of role and content of response.
        """

        def handle_search_text_chat_completions(data: dict) -> Optional[tuple[Role, Content]]:
            """
            Internal method, describes how RegoloClient.create_stream_generator() should handle
            output from chat_completions.
            """
            if isinstance(data, dict):
                delta = data.get("choices", [{}])[0].get("delta", {})
                out_role: Role = delta.get("role", "")
                out_content: Content = delta.get("content", "")
                return out_role, out_content
            elif isinstance(data, list):
                for element in data:
                    delta = element.get("choices", [{}])[0].get("delta", {})
                    out_role: Role = delta.get("role", "")
                    out_content: Content = delta.get("content", "")
                    return out_role, out_content
            return None

        # Use the default API key if not provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate the API key
        api_key = KeysHandler.check_key(api_key)

        # Use the default model if not specified
        if model is None:
            model = regolo.default_chat_model

        # Validate the model
        ModelsHandler.check_model(model=model, base_url=base_url, api_key=api_key)

        # Convert the Conversation object to a list of message dictionaries
        if type(messages) == Conversation:
            messages = messages.get_lines()

        # Create a new HTTP client if one is not provided
        if client is None:
            client = httpx.Client()

        # Construct the payload for the API request
        payload = {
            "model": model,
            "stream": stream,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        # Remove None values from payload to avoid unnecessary parameters
        payload = {k: v for k, v in payload.items() if v is not None}

        # Set authorization header
        headers = {"Authorization": api_key}

        if stream:
            # If streaming, return a generator for handling the response
            return RegoloClient.create_stream_generator(
                client=client,
                base_url=base_url,
                payload=payload,
                headers=headers,
                full_output=full_output,
                search_url=os.getenv("CHAT_COMPLETIONS_URL_PATH"),
                output_handler=handle_search_text_chat_completions
            )
        else:
            # Send a synchronous POST request
            response = safe_post(
                client=client,
                url_to_query=f"{base_url}{os.getenv("CHAT_COMPLETIONS_URL_PATH")}",
                json_to_query=payload,
                headers_to_query=headers
            ).json()

            if full_output:
                # Return full response if requested
                return response
            else:
                # Extract role and content from response
                role = response["choices"][0]["message"]["role"]
                content = response["choices"][0]["message"]["content"]
                return role, content

    def add_prompt_to_chat(self, prompt: str, role: str):
        """
        Adds a prompt to the chat as the role specified

        :param prompt: The prompt to add.
        :param role: The role of the prompt to add.

        :example:
            client = RegoloClient()
            client.add_prompt_to_chat(prompt="how are you?", role="user")
            print(client.run_chat())
        """

        self.instance.add_prompt_as_role(prompt=prompt, role=role)

    def clear_conversations(self) -> None:
        """clear all prompts to start new conversations."""
        self.instance.clear_conversation()

    def run_chat(self,
                 user_prompt: Optional[str] = None,
                 stream: bool = False,
                 max_tokens: int = 200,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 full_output: bool = False) -> GeneratorType | tuple[Role, Content]:
        """
        Runs chat endpoint from RegoloClient instance (self.conversation contains the role-prompts dicts).

        :param user_prompt: Optional prompt to add to conversation before generating the response from regolo.ai.
        :param stream: Whether to stream the prompt from regolo.ai.
        :param max_tokens: Maximum number of tokens to generate. (Defaults to 200)
        :param temperature: Sampling temperature for randomness.
        :param top_p: Nucleus sampling parameter.
        :param top_k: Top-k sampling parameter.
        :param full_output: Whether to return full response. (Defaults to False)
        :return for stream=true, full_output=False: Generator, which yields dicts with the responses from regolo.ai.
        :return for stream=True, full_output=True: Generator, which yields tuples of Role, Content of response.
        :return for stream=False, full_output=False: String, with response from regolo.ai.
        :return for stream=False, full_output=True: Tuple, which consists of role and content of response.
        """

        if user_prompt is not None:
            self.instance.add_prompt_as_role(prompt=user_prompt, role="user")

        if self.instance.get_base_url() is None:
            base_url = os.getenv("REGOLO_URL")
        else:
            base_url = self.instance.get_base_url()

        response = self.static_chat_completions(messages=self.instance.get_conversation(),
                                                model=self.instance.get_model(),
                                                stream=stream,
                                                api_key=self.instance.get_api_key(),
                                                max_tokens=max_tokens,
                                                temperature=temperature,
                                                top_p=top_p,
                                                top_k=top_k,
                                                client=self.instance.get_client(),
                                                base_url=base_url,
                                                full_output=full_output)

        if stream:
            return response
        else:
            if full_output:
                responseRole = response["choices"][0]["message"]["role"]
                responseText = response["choices"][0]["message"]["content"]
            else:
                responseRole = response[0]
                responseText = response[1]

            self.instance.add_line(ConversationLine(role=responseRole, content=responseText))

            return response

    # Create images
    @staticmethod
    def static_create_image(prompt: str,
                            model: Optional[str] = None,
                            api_key: Optional[str] = None,
                            n: int = 1,
                            quality: str = "standard",
                            size: str = "1024x1024",
                            style: str = "realistic",
                            client: Optional[httpx.Client] = None,
                            base_url: str = os.getenv("REGOLO_URL"),
                            full_output: bool = False) -> list[bytes] | dict:
        """
        Generates an image based on the given prompt using the regolo.ai image model.

        :param prompt: The text prompt for image generation.
        :param model: The regolo.ai image model to use. (Optional)
        :param api_key: The API key for regolo.ai. (Optional)
        :param n: The number of images to generate. (Defaults to 1)
        :param quality: The quality of the image that will be generated. The "hd" value creates images with finer details and greater consistency across the image. (Defaults to "standard")
        :param size: The size of the generated images.
        :param style: The style of the generated images. (Defaults to "realistic")
        :param client: The HTTP client for making requests. (Optional)
        :param base_url: Base URL of the regolo HTTP server. (Defaults to REGOLO_URL)
        :param full_output: Whether to return full response. (Defaults to False)

        :return full_output=True: Dict containing the text of the response.
        :return full_output=False: List containing the images decoded as bytes.
        """

        # Use the default API key if not provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate the API key
        api_key = KeysHandler.check_key(api_key)

        # Use the default model if not specified
        if model is None:
            model = regolo.default_image_generation_model

        # Validate the model
        ModelsHandler.check_model(model=model, base_url=base_url, api_key=api_key)

        # Create a new HTTP client if one is not provided
        if client is None:
            client = httpx.Client()

        # Construct the payload for the API request
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "quality": quality,
            "size": size,
            "style": style
        }

        # Remove None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        # Set authorization header
        headers = {"Authorization": api_key}

        # Send a synchronous POST request
        response = safe_post(
            client=client,
            url_to_query=f"{base_url}{os.getenv("IMAGE_GENERATION_URL_PATH")}",
            json_to_query=payload,
            headers_to_query=headers
        ).json()

        if full_output:
            return response
        else:
            # Extract the image URL from response
            return [b64decode(img_info["b64_json"]) for img_info in response["data"]]

    def create_image(self,
                     prompt: str,
                     n: int = 1,
                     quality: str = "standard",
                     size: str = "1024x1024",
                     style: str = "realistic",
                     full_output: bool = False) -> list[bytes] | dict:
        """
        Generates an image based on the given prompt using the regolo.ai image model.

        :param prompt: The text prompt for image generation.
        :param n: The number of images to generate. (Defaults to 1)
        :param quality: The quality of the image that will be generated.
            The "hd" value creates images with finer details and greater consistency across the image.
            (Defaults to "standard")
        :param size: The size of the generated images.
        :param style: The style of the generated images. (Defaults to "vivid")
        :param full_output: Whether to return full response. (Defaults to False)

        :return full_output=True: Dict containing the text of the response.
        :return full_output=False: List containing the images decoded as bytes.
        """

        if self.instance.get_base_url() is None:
            base_url = os.getenv("REGOLO_URL")
        else:
            base_url = self.instance.get_base_url()

        response = self.static_create_image(prompt=prompt,
                                            model=self.instance.get_image_model(),
                                            api_key=self.instance.get_api_key(),
                                            n=n,
                                            quality=quality,
                                            size=size,
                                            style=style,
                                            client=self.instance.get_client(),
                                            base_url=base_url,
                                            full_output=full_output)

        return response

    # Generate embeddings
    @staticmethod
    def static_embeddings(input_text: list[str] | str,
                          model: Optional[str] = None,
                          api_key: Optional[str] = None,
                          client: Optional[httpx.Client] = None,
                          base_url: str = os.getenv("REGOLO_URL"),
                          full_output: bool = False) -> dict | list:
        """

        :param input_text: The text to be embedded.
        :param model: The regolo.ai image model to use. (Optional)
        :param api_key: The API key for regolo.ai. (Optional)
        :param client: HTTP client for making requests. (Optional)
        :param base_url: Base URL of the regolo HTTP server. (Optional)
        :param full_output: Whether to return full response. (Defaults to False)
        """
        # Use the default API key if none is provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        if model is None:
            model = regolo.default_embedder_model

        # Validate the selected model
        ModelsHandler.check_model(model=model, api_key=api_key, base_url=base_url)

        # Create a new HTTP client if none is provided
        if client is None:
            client = httpx.Client()

        # Construct the payload for the API request
        payload = {
            "input": input_text,
            "model": model,
        }

        # Remove None values from payload to avoid unnecessary parameters
        payload = {k: v for k, v in payload.items() if v is not None}

        # Set authorization header
        headers = {"Authorization": api_key}

        response = safe_post(
            client=client,
            url_to_query=f"{base_url}{os.getenv("EMBEDDINGS_URL_PATH")}",
            json_to_query=payload,
            headers_to_query=headers
        )

        if full_output:
            return response.json()
        else:
            return response.json()["data"]

    def embeddings(self,
                   input_text: list[str] | str,
                   full_output: bool = False) -> dict | list:
        """
        :param input_text: The text to be embedded.
        :param full_output: Whether to return full response. (Defaults to False)
        """
        if self.instance.get_base_url() is None:
            base_url = os.getenv("REGOLO_URL")
        else:
            base_url = self.instance.get_base_url()

        return self.static_embeddings(input_text=input_text,
                                      full_output=full_output,
                                      model=self.instance.get_embedder_model(),
                                      api_key=self.instance.get_api_key(),
                                      client=self.instance.get_client(),
                                      base_url=base_url)

    @staticmethod
    def static_audio_transcription(file,
                                   model: Optional[str] = None,
                                   api_key: Optional[str] = None,
                                   chunking_strategy: Optional[str | dict] = None,
                                   include: Optional[List[str]] = None,
                                   language: Optional[str] = None,
                                   prompt: Optional[str] = None,
                                   response_format: str = "json",
                                   stream: bool = False,
                                   temperature: Optional[float] = 0,
                                   timestamp_granularities: Optional[List[str]] = None,
                                   client: Optional[httpx.Client] = None,
                                   base_url: str = os.getenv("REGOLO_URL"),
                                   full_output: bool = False) -> str | dict | Generator[Any, Any, None]:
        """
        Transcribes audio using the regolo.ai audio transcription model.

        :param file: The audio file object (bytes, file-like an object, or path string) to transcribe,
            in formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
        :param model: The name of the model to use.
            Example: faster-whisper-large-v3
            (Optional)
        :param api_key: The API key for regolo.ai.
            (Optional)
        :param chunking_strategy: Controls how audio is cut into chunks.
            Auto or object.
            (Optional)
        :param include: Additional information to include in response.
            (Optional)
        :param language: The language of the input audio in ISO-639-1 format.
            (Optional)
        :param prompt: An optional text to guide the model's style or continue a previous audio segment.
            (Optional)
        :param response_format: The format of the output: json, text, srt, verbose_json, or vtt.
            (Defaults to "json")
        :param stream: If true, stream the response using server-sent events.
            Note: Not supported for whisper-1.
            (Defaults to False)
        :param temperature: The sampling temperature, between 0 and 1.
            (Defaults to 0)
        :param timestamp_granularities: Timestamp granularities: word or segment.
            Requires verbose_json format.
            (Optional)
        :param client: httpx client to use.
            (Optional)
        :param base_url: Base URL of the regolo HTTP server.
            (Defaults to REGOLO_URL)
        :param full_output: Whether to return the full response.
            (Defaults to False)

        :return for stream=True: Generator yielding streaming responses.
        :return for stream=False, full_output=True: Dict containing the full response from regolo.ai.
        :return for stream=False, full_output=False: String with transcribed text.
        """

        def handle_search_audio_transcription(output_data: dict) -> Optional[str]:
            """
            Internal method, describes how RegoloClient.create_stream_generator() should handle
            output from audio transcriptions.
            """
            if isinstance(output_data, dict):
                # Extract text content from streaming chunks
                return output_data.get("text", "")
            return ""

        # Use the default API key if none is provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        if model is None:
            model = regolo.default_audio_transcription_model

        # Validate the selected model
        ModelsHandler.check_model(model=model, api_key=api_key, base_url=base_url)

        # Create a new HTTP client if none is provided
        if client is None:
            client = httpx.Client()

        # Handle different file input types
        try:
            if isinstance(file, str):
                # File path provided
                with open(file, 'rb') as audio_file:
                    file_content = audio_file.read()
                    file_name = os.path.basename(file)
            elif isinstance(file, bytes):
                # Bytes provided
                file_content = file
                file_name = "audio_file"
            elif hasattr(file, 'read'):
                # File-like object provided
                file_content = file.read()
                file_name = getattr(file, 'name', 'audio_file')
            else:
                raise ValueError("File must be a path string, bytes, or file-like object")

            files = {
                'file': (file_name, file_content, 'audio/*')
            }

            # Construct the data payload (form data, not JSON for file uploads)
            data = {
                'model': model,
                'response_format': response_format,
                'stream': str(stream).lower(),
            }

            # Add optional parameters if provided
            if chunking_strategy is not None:
                if isinstance(chunking_strategy, str):
                    data['chunking_strategy'] = chunking_strategy
                else:
                    # Handle object case - convert to JSON string
                    data['chunking_strategy'] = json.dumps(chunking_strategy)

            if include is not None:
                for item in include:
                    data[f'include[]'] = item

            if language is not None:
                data['language'] = language
            if prompt is not None:
                data['prompt'] = prompt
            if temperature is not None:
                data['temperature'] = str(temperature)
            if timestamp_granularities is not None:
                for granularity in timestamp_granularities:
                    data[f'timestamp_granularities[]'] = granularity

            # Set authorization header
            headers = {"Authorization": api_key}

            if stream:
                # Use the generalized create_stream_generator for streaming
                return RegoloClient.create_stream_generator(
                    client=client,
                    base_url=base_url,
                    files=files,
                    data=data,
                    headers=headers,
                    full_output=full_output,
                    search_url=os.getenv("AUDIO_TRANSCRIPTION_URL_PATH"),
                    output_handler=handle_search_audio_transcription
                )

            # Non-streaming case
            response = client.post(
                url=f"{base_url}{os.getenv('AUDIO_TRANSCRIPTION_URL_PATH')}",
                files=files,
                data=data,
                headers=headers,
                timeout=timeout
            )
            response.raise_for_status()

        except FileNotFoundError:
            raise FileNotFoundError(f"Audio file not found: {file}")
        except Exception as e:
            raise Exception(f"Error processing audio file: {str(e)}")

        response_json = response.json()

        if full_output:
            return response_json
        else:
            # Return just the transcribed text
            return response_json.get("text", "")

    def audio_transcription(self,
                            file,
                            model: Optional[str] = None,
                            chunking_strategy: Optional[str | dict] = None,
                            include: Optional[List[str]] = None,
                            language: Optional[str] = None,
                            prompt: Optional[str] = None,
                            response_format: str = "json",
                            stream: bool = False,
                            temperature: Optional[float] = 0,
                            timestamp_granularities: Optional[List[str]] = None,
                            full_output: bool = False) -> str | dict | GeneratorType:
        """
        Transcribes audio using the regolo.ai audio transcription model from RegoloClient instance.

        :param file: The audio file object (bytes, file-like an object, or path string) to transcribe,
            in formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.
        :param model: The name of the model to use.
            Example: faster-whisper-large-v3
            (Optional)
        :param chunking_strategy: Controls how audio is cut into chunks.
            Auto or object.
            (Optional)
        :param include: Additional information to include in response.
            (Optional)
        :param language: The language of the input audio in ISO-639-1 format.
            (Optional)
        :param prompt: An optional text to guide the model's style or continue a previous audio segment.
            (Optional)
        :param response_format: The format of the output: json, text, srt, verbose_json, or vtt.
            (Defaults to "json")
        :param stream: If true, stream the response using server-sent events.
            Note: Not supported for whisper-1.
            (Defaults to False)
        :param temperature: The sampling temperature, between 0 and 1.
            (Defaults to 0)
        :param timestamp_granularities: Timestamp granularities: word or segment.
            Requires verbose_json format.
            (Optional)
        :param full_output: Whether to return the full response.
            (Defaults to False)

        :return for stream=True: Generator yielding streaming responses.
        :return for stream=False, full_output=True: Dict containing the full response from regolo.ai.
        :return for stream=False, full_output=False: String with transcribed text.
        """

        if self.instance.get_base_url() is None:
            base_url = os.getenv("REGOLO_URL")
        else:
            base_url = self.instance.get_base_url()

        # Use the instance model if not specified
        if model is None:
            model = self.instance.get_model()

        return self.static_audio_transcription(
            file=file,
            model=model,
            api_key=self.instance.get_api_key(),
            chunking_strategy=chunking_strategy,
            include=include,
            language=language,
            prompt=prompt,
            response_format=response_format,
            stream=stream,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            client=self.instance.get_client(),
            base_url=base_url,
            full_output=full_output
        )

    @staticmethod
    def static_rerank(query: str,
                      documents: List[str] | List[Dict[str, Any]],
                      model: Optional[str] = None,
                      api_key: Optional[str] = None,
                      top_n: Optional[int] = None,
                      rank_fields: Optional[List[str]] = None,
                      return_documents: bool = True,
                      max_chunks_per_doc: Optional[int] = None,
                      client: Optional[httpx.Client] = None,
                      base_url: str = os.getenv("REGOLO_URL"),
                      full_output: bool = False) -> List[Dict[str, Any]] | dict:
        """
        Reranks a list of documents based on their relevance to a query using regolo.ai reranking model.

        :param query: The search query to compare documents against.
        :param documents: List of documents to rerank.
        Can be strings or dicts with text fields.
        :param model: The regolo.ai reranking model to use.
        (Optional)
        :param api_key: The API key for regolo.ai.
        (Optional)
        :param top_n: Number of most relevant documents to return.
        Returns all if not specified.
        (Optional)
        :param rank_fields: For structured documents, specify which fields to rank by.
        (Optional)
        :param return_documents: Whether to return document content in results.
        (Defaults to True)
        :param max_chunks_per_doc: Maximum number of chunks per document.
        (Optional)
        :param client: httpx client to use.
        (Optional)
        :param base_url: Base URL of the regolo HTTP server.
        (Defaults to REGOLO_URL)
        :param full_output: Whether to return the full response.
        (Defaults to False)

        :return full_output=True: Dict containing the full response from regolo.ai.
        :return full_output=False: List of dicts with 'index', 'relevance_score', and optionally 'document'.
        """

        # Use the default API key if none is provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        # Use the default reranking model if not specified
        if model is None:
            model = regolo.default_reranker_model

        # Validate the selected model
        ModelsHandler.check_model(model=model, api_key=api_key, base_url=base_url)

        # Create a new HTTP client if none is provided
        if client is None:
            client = httpx.Client()

        # Construct the payload for the API request
        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "rank_fields": rank_fields,
            "return_documents": return_documents,
            "max_chunks_per_doc": max_chunks_per_doc
        }

        # Remove None values from payload to avoid unnecessary parameters
        payload = {k: v for k, v in payload.items() if v is not None}

        # Set authorization header
        headers = {"Authorization": api_key}

        # Send a synchronous POST request
        response = safe_post(
            client=client,
            url_to_query=f"{base_url}{os.getenv('RERANK_URL_PATH')}",
            json_to_query=payload,
            headers_to_query=headers
        )

        response_json = response.json()

        if full_output:
            return response_json
        else:
            # Return just the ranked results
            return response_json.get("results", [])

    def rerank(self,
               query: str,
               documents: List[str] | List[Dict[str, Any]],
               top_n: Optional[int] = None,
               rank_fields: Optional[List[str]] = None,
               return_documents: bool = True,
               max_chunks_per_doc: Optional[int] = None,
               full_output: bool = False) -> List[Dict[str, Any]] | dict:
        """
        Reranks a list of documents based on their relevance to a query using regolo.ai reranking model
        from RegoloClient instance.

        :param query: The search query to compare documents against.
        :param documents: List of documents to rerank.
        Can be strings or dicts with text fields.
        :param top_n: Number of most relevant documents to return.
        Returns all if not specified.
        (Optional)
        :param rank_fields: For structured documents, specify which fields to rank by.
        (Optional)
        :param return_documents: Whether to return document content in results.
        (Defaults to True)
        :param max_chunks_per_doc: Maximum number of chunks per document.
        (Optional)
        :param full_output: Whether to return the full response.
        (Defaults to False)

        :return full_output=True: Dict containing the full response from regolo.ai.
        :return full_output=False: List of dicts with 'index', 'relevance_score', and optionally 'document'.
        """

        if self.instance.get_base_url() is None:
            base_url = os.getenv("REGOLO_URL")
        else:
            base_url = self.instance.get_base_url()

        return self.static_rerank(
            query=query,
            documents=documents,
            model=self.instance.get_reranker_model(),
            api_key=self.instance.get_api_key(),
            top_n=top_n,
            rank_fields=rank_fields,
            return_documents=return_documents,
            max_chunks_per_doc=max_chunks_per_doc,
            client=self.instance.get_client(),
            base_url=base_url,
            full_output=full_output
        )
