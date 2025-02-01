import json
from types import GeneratorType

import httpx
from typing import Any, Dict, List, Optional

from json_repair import json_repair

from regolo.instance.regolo_instance import RegoloInstance
from regolo.instance.structures.conversation_model import Conversation, ConversationLine
from regolo.keys.keys import KeysHandler
from regolo.models.models import ModelsHandler
from typing import TypeAlias

import regolo

REGOLO_URL = "https://api.regolo.ai"
COMPLETIONS_URL_PATH = "/v1/completions"
CHAT_COMPLETIONS_URL_PATH = "/v1/chat/completions"
timeout = 500

Role: TypeAlias = str
Content: TypeAlias = str

def safe_post(client: httpx.Client,
              url_to_query: str,
              json_to_query: dict = None,
              headers_to_query: dict = None) -> httpx.Response:
    response = client.post(url=url_to_query, json=json_to_query, headers=headers_to_query, timeout=timeout)
    response.raise_for_status()
    return response


class RegoloClient:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 alternative_url: Optional[str] = None,
                 pre_existent_conversation: Optional[Conversation] = None,
                 pre_existent_client: httpx.Client = None) -> None:
        """
        Initialize the client for vLLM HTTP API.

        Args:
            alternative_url (str): Base URL of the regolo HTTP server.
            pre_existent_conversation (RegoloInstance): eventual conversation to start chatting with.
        """
        self.base_url = REGOLO_URL if alternative_url is None else alternative_url
        client = httpx.Client(base_url=self.base_url) if pre_existent_client is None else pre_existent_client
        self.instance = RegoloInstance(model=regolo.default_model if model is None else model,
                                       api_key=regolo.default_key if api_key is None else api_key,
                                       previous_conversations=pre_existent_conversation, client=client)

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
    def get_available_models() -> List[str]:
        """Gets all available models on regolo.ai."""
        return ModelsHandler.get_models()

    @staticmethod
    def process_vllm_stream(tokens):
        output_text = ""
        for token in tokens:
            if token.startswith("▁"):  # SentencePiece-style tokenization
                output_text += " " + token[1:]
            else:
                output_text += token  # Continuation of the previous word
        return output_text.strip()

    @staticmethod
    def create_stream_generator(client: httpx.Client, base_url: str, payload: dict, headers: dict,
                                full_output: bool, search_url, output_handler) -> GeneratorType:

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
                           stream: Optional[bool] = False,
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None,
                           top_p: Optional[float] = None,
                           top_k: Optional[int] = None,
                           client: Optional[httpx.Client] = None,
                           base_url: str = REGOLO_URL,
                           full_output: Optional[bool] = False) -> GeneratorType:
        """Will return generators for stream=True and values for stream=False
        Send a prompt to regolo server and get the generated response.

        Args:
            prompt (str): The input prompt to the LLM.
            model (str): The regolo.ai model to use.
            api_key (str): The API key for regolo.ai.
            stream (bool): Whether to stream the prompt from regolo.ai.
            max_tokens (int, optional): Maximum number of tokens to generate.
            temperature (float, optional): Sampling temperature for randomness.
            top_p (float, optional): Nucleus sampling parameter.
            top_k (int, optional): Top-k sampling parameter.
            client (httpx.Client, optional): httpx client to use.
            base_url (str): Base URL of the regolo HTTP server.
            full_output (bool, optional): Whether to return full response.

        Returns:
            dict: Response from the vLLM server.
ì        """

        def handle_search_text_completions(data: dict):
            return data.get("choices", [{}])[0].get("text")

        if api_key is None:
            api_key = regolo.default_key
        if model is None:
            model = regolo.default_model
        api_key = KeysHandler.check_key(api_key)

        if client is None:
            client = httpx.Client()
        payload = {
            "prompt": prompt,
            "stream": stream,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        # Filter out parameters that are None
        payload = {k: v for k, v in payload.items() if v is not None}
        headers = {"Authorization": api_key}
        if stream:
            return RegoloClient.create_stream_generator(client=client, base_url=base_url, payload=payload,
                                                        headers=headers,
                                                        full_output=full_output, search_url=COMPLETIONS_URL_PATH,
                                                        output_handler=handle_search_text_completions)
        else:
            response = safe_post(client=client, url_to_query=f"{base_url}{COMPLETIONS_URL_PATH}",
                                 json_to_query=payload, headers_to_query=headers)
            if full_output:
                return response.json()
            else:
                return response.json()["choices"][0]["text"]

    def completions(self,
                    prompt: str,
                    stream: Optional[bool] = False,
                    max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None,
                    top_p: Optional[float] = None,
                    top_k: Optional[int] = None,
                    full_output: bool = False) -> Dict[str, Any] | GeneratorType:

        """Performs requests to completions endpoint from RegoloClient instance."""

        response = self.static_completions(prompt=prompt,
                                           model=self.instance.get_model(),
                                           api_key=self.instance.get_api_key(),
                                           stream=stream,
                                           max_tokens=max_tokens,
                                           temperature=temperature,
                                           top_p=top_p,
                                           top_k=top_k,
                                           client=self.instance.client,
                                           base_url=self.base_url,
                                           full_output=full_output)

        return response

    # Chat completions

    @staticmethod
    def static_chat_completions(messages: Conversation | List[Dict[str, str]],
                                model: Optional[str] = regolo.default_model,
                                api_key: Optional[str] = regolo.default_key,
                                stream: bool = False,
                                max_tokens: Optional[int] = None,
                                temperature: Optional[float] = None,
                                top_p: Optional[float] = None,
                                top_k: Optional[int] = None,
                                client: Optional[httpx.Client] = None,
                                base_url: str = REGOLO_URL,
                                full_output: bool = False
                                ) -> GeneratorType | tuple[Role, Content] | dict:
        """
        Internal method, returns generators.
        Sends a series of chat messages to the vLLM server and gets the response.

        Args:
            messages (List[Dict[str, str]]): A list of messages in the format ["role": "user"|"assistant", "content": "message"].
            model (str): The regolo.ai model to use.
            api_key (str): The API key for regolo.ai.
            stream (bool): Whether to stream the prompt from regolo.ai.
            max_tokens (int, optional): Maximum number of tokens to generate.
            temperature (float, optional): Sampling temperature for randomness.
            top_p (float, optional): Nucleus sampling parameter.
            top_k (int, optional): Top-k sampling parameter.
            client (httpx.Client): httpx client to use.
            base_url (str): Base URL of the regolo HTTP server.
            full_output (bool, optional): Whether to return full response.

        Returns:
            dict: Response from the vLLM server.
        """

        def handle_search_text_chat_completions(data: dict) -> tuple[Role, Content]:
            if isinstance(data, dict):
                delta = data.get("choices", [{}])[0].get("delta", {})
                out_role = delta.get("role", "")
                out_content = delta.get("content", "")
                return out_role, out_content
            elif isinstance(data, list):
                for element in data:
                    delta = element.get("choices", [{}])[0].get("delta", {})
                    out_role = delta.get("role", "")
                    out_content = delta.get("content", "")
                    return out_role, out_content

        if type(messages) == Conversation:
            messages = messages.get_lines()  # TODO: to test
        api_key = KeysHandler.check_key(api_key)

        if client is None:
            client = httpx.Client()

        payload = {
            "model": model,
            "stream": stream,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        # Filter out parameters that are None
        payload = {k: v for k, v in payload.items() if v is not None}
        headers = {"Authorization": api_key}

        if stream:
            return RegoloClient.create_stream_generator(client=client, base_url=base_url, payload=payload,
                                                        headers=headers,
                                                        full_output=full_output, search_url=CHAT_COMPLETIONS_URL_PATH,
                                                        output_handler=handle_search_text_chat_completions)
        else:
            response = safe_post(client=client, url_to_query=f"{base_url}{CHAT_COMPLETIONS_URL_PATH}",
                                 json_to_query=payload, headers_to_query=headers).json()
            if full_output:
                return response
            else:
                role = response["choices"][0]["message"]["content"]
                content = response["choices"][0]["message"]["content"]
                return role, content


    def add_prompt_to_chat(self, prompt: str, role):
        """
        Adds a prompt to the chat as the role specified

        Args:
            prompt (str): The prompt to add.
            role (str): The role of the prompt to add.

        Example usage:
            client = RegoloClient()
            client.add_prompt_to_chat(prompt="how are you?", role="user")
            client.run()
        """
        self.instance.add_prompt_as_role(prompt=prompt, role=role)

    def clear_conversations(self) -> None:
        """clear all prompts to start new conversations."""
        self.instance.clear_conversation()

    def run_chat(self,
                 user_prompt: Optional[str] = None,
                 stream: bool = False,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_p: Optional[float] = None,
                 top_k: Optional[int] = None,
                 full_output: bool = False) -> GeneratorType | tuple[Role, Content]:
        if user_prompt is not None:
            self.instance.add_prompt_as_role(prompt=user_prompt, role="user")

        response = self.static_chat_completions(messages=self.instance.get_conversation(),
                                                model=self.instance.get_model(),
                                                stream=stream,
                                                api_key=self.instance.get_api_key(),
                                                max_tokens=max_tokens,
                                                temperature=temperature,
                                                top_p=top_p,
                                                top_k=top_k,
                                                client=self.instance.get_client(),
                                                base_url=self.base_url,
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
