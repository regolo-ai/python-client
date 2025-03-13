import json
import os
from base64 import b64decode
from types import GeneratorType
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generator
from typing import List
from typing import Optional
from typing import TypeAlias

import httpx
from json_repair import json_repair

import regolo
from regolo.instance.regolo_instance import RegoloInstance
from regolo.instance.structures.conversation_model import Conversation
from regolo.instance.structures.conversation_model import ConversationLine
from regolo.keys.keys import KeysHandler
from regolo.models.models import ModelsHandler
from dotenv import load_dotenv

load_dotenv(override=False)

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
                 model: Optional[str] = None,
                 embedder_model: Optional[str] = None,
                 image_model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 alternative_url: Optional[str] = None,
                 pre_existent_conversation: Optional[Conversation] = None,
                 pre_existent_client: httpx.Client = None) -> None:
        """
        Initialize the client for regolo.ai HTTP API.

        :param model: The regolo.ai model to use. (Defaults to regolo.default_model)
        :param api_key: The API key for regolo.ai. (Defaults to regolo.default_key)
        :param alternative_url: Base URL of the regolo HTTP server. (Optional)
        :param pre_existent_conversation: An existing conversation instance to continue chatting with. (Optional)
        :param pre_existent_client: An existing httpx.Client instance to use. (Optional)
        """

        model = regolo.default_model if model is None else model
        embedder_model = regolo.default_embedder_model if embedder_model is None else embedder_model
        image_model = regolo.default_image_model if image_model is None else image_model
        api_key = regolo.default_key if api_key is None else api_key
        base_url = None if alternative_url is None else alternative_url
        client = httpx.Client(base_url=os.getenv("REGOLO_URL") if base_url is None else base_url) if pre_existent_client is None else pre_existent_client

        self.instance = RegoloInstance(model=model,
                                       embedder_model=embedder_model,
                                       image_model=image_model,
                                       api_key=api_key,
                                       previous_conversations=pre_existent_conversation, client=client,
                                       base_url=base_url)

    @classmethod
    def from_instance(cls, instance: RegoloInstance, alternative_url: Optional[str] = None) -> "RegoloClient":
        """Creates RegoloClient from instance."""
        return cls(api_key=instance.api_key, model=instance.model, alternative_url=alternative_url,
                   pre_existent_client=instance.client, pre_existent_conversation=instance.conversation)

    def change_model(self, model: str) -> None:
        """Change model used in this instance of regolo_client"""
        try:
            self.instance.change_model(new_model=model)
        except Exception as e:
            print(e)

    @staticmethod
    def get_available_models(api_key: str, base_url: str=os.getenv("REGOLO_URL")) -> List[str]:
        """
        Gets all available models on regolo.ai.

        :param base_url: Base URL of the regolo HTTP server.
        :param api_key: The API key for regolo.ai.

        :return: A list of available models.
        """

        # Validate API key
        api_key = KeysHandler.check_key(api_key)

        return ModelsHandler.get_models(base_url=base_url, api_key=api_key)

    @staticmethod
    def create_stream_generator(client: httpx.Client,
                                base_url: str,
                                payload: dict,
                                headers: dict,
                                full_output: bool,
                                search_url: str,
                                output_handler: Callable[[Dict], Any]) -> Generator[Any, Any, None]:
        """
        Yields generators for streams from regolo.ai.

        :param client: The httpx.Client instance to use.
        :param base_url: Base URL of the regolo HTTP server.
        :param payload: The request payload to send.
        :param headers: The request headers.
        :param full_output: Whether to return the full response.
        :param search_url: The URL for the search request.
        :param output_handler: A function that processes responses if full_output=False.
        :return: A generator that yields streamed responses from regolo.ai.
        """
        with client.stream("POST", f"{base_url}{search_url}", json=payload, headers=headers) as response:
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
                    data = json.loads(json_repair.repair_json(decoded_line))
                except (Exception,):
                    continue

                if full_output:
                    yield data
                else:
                    # Handle both dict and list responses uniformly
                    yield output_handler(data)

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
            model = regolo.default_model

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

        # Use the default API key if not provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate the API key
        api_key = KeysHandler.check_key(api_key)

        # Use the default model if not specified
        if model is None:
            model = regolo.default_model

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

        :param user_prompt: Optional prompt to add to conversation before generating response from regolo.ai.
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

        if stream is True:
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
        :param style: The style of the generated images. (Defaults to "vivid")
        :param client: The HTTP client for making requests. (Optional)
        :param base_url: Base URL of the regolo HTTP server. (Defaults to REGOLO_URL)
        :param full_output: Whether to return full response. (Defaults to False)

        :return full_output=True: Dict containing the text of response.
        :return full_output=False: List containing the images decoded as bytes.
        """

        # Use the default API key if not provided
        if api_key is None:
            api_key = regolo.default_key

        # Validate the API key
        api_key = KeysHandler.check_key(api_key)

        # Use the default model if not specified
        if model is None:
            model = regolo.default_image_model

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
        :param quality: The quality of the image that will be generated. The "hd" value creates images with finer details and greater consistency across the image. (Defaults to "standard")
        :param size: The size of the generated images.
        :param style: The style of the generated images. (Defaults to "vivid")
        :param full_output: Whether to return full response. (Defaults to False)

        :return full_output=True: Dict containing the text of response.
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
