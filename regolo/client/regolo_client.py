import httpx
from typing import Any, Dict, List, Optional, Type

from regolo.instance.regolo_instance import RegoloInstance
from regolo.instance.structures.conversation_model import Conversation, ConversationLine
from regolo.keys.keys import KeysHandler
from regolo.models.models import ModelsHandler

import regolo

REGOLO_URL = "https://api.regolo.ai"
timeout = 500

def safe_post(client: httpx.Client, url: str, json: dict = None, headers: dict = None) -> httpx.Response:
    response = client.post(url=url, json=json, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


class RegoloClient:
    def __init__(self, model: Optional[str]=None, api_key: Optional[str]=None, alternative_url: Optional[str] = None,
                 pre_existent_conversation: Optional[Conversation] = None, pre_existent_client: httpx.Client = None) -> None:
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
        return cls(api_key=instance.api_key, model=instance.model, alternative_url=alternative_url,
                   pre_existent_client=instance.client, pre_existent_conversation=instance.conversation)

    def change_model(self, model:str) -> None:
        try:
            self.instance.change_model(new_model=model)
        except Exception as e:
            print(e)

    @staticmethod
    def get_available_models() -> List[str]:
        return ModelsHandler.get_models()

    @staticmethod
    def static_completions(prompt: str,
                           model: Optional[str]=None,
                           api_key: Optional[str]=None,
                           stream: Optional[bool] = False,
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None,
                           top_p: Optional[float] = None,
                           top_k: Optional[int] = None,
                           client: Optional[httpx.Client] = None,
                           base_url: str = REGOLO_URL) -> Dict[str, Any]:
        """
        Send a prompt to regolo server and get the generated response.

        Args:
            prompt (str): The input prompt to the LLM.
            model (str): The regolo.ai model to use.
            api_key (str): The API key for regolo.ai.
            max_tokens (int, optional): Maximum number of tokens to generate.
            temperature (float, optional): Sampling temperature for randomness.
            top_p (float, optional): Nucleus sampling parameter.
            top_k (int, optional): Top-k sampling parameter.
            client (httpx.Client, optional): httpx client to use.
            base_url (str): Base URL of the regolo HTTP server.

        Returns:
            dict: Response from the vLLM server.
        """
        if api_key is None:
            api_key = regolo.default_key
        if model is None:
            model = regolo.default_model
        api_key = KeysHandler.check_key(api_key)

        if client is None:
            client = httpx.Client()
        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }

        # Filter out parameters that are None
        payload = {k: v for k, v in payload.items() if v is not None}
        headers = {"Authorization": api_key}

        response = safe_post(client=client, url=f"{base_url}/v1/completions", json=payload, headers=headers)
        return response.json()

    def completions(self,
                    prompt: str,
                    max_tokens: Optional[int] = None,
                    temperature: Optional[float] = None,
                    top_p: Optional[float] = None,
                    top_k: Optional[int] = None,
                    full_output: bool = False) -> Dict[str, Any]:
        response = self.static_completions(prompt=prompt,
                                           model=self.instance.get_model(),
                                           api_key=self.instance.get_api_key(),
                                           max_tokens=max_tokens,
                                           temperature=temperature,
                                           top_p=top_p,
                                           top_k=top_k,
                                           client=self.instance.client,
                                           base_url=self.base_url)

        if full_output:
            return response
        else:
            return response["choices"][0]["text"]

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
                                base_url: str = REGOLO_URL
                                ) -> Dict[str, Any]:
        """
        Send a series of chat messages to the vLLM server and get the response.

        Args:
            messages (List[Dict[str, str]]): A list of messages in the format ["role": "user"|"assistant", "content": "message"].
            model (str): The regolo.ai model to use.
            stream (bool, optional): Whether to stream messages.
            api_key (str): The API key for regolo.ai.
            max_tokens (int, optional): Maximum number of tokens to generate.
            temperature (float, optional): Sampling temperature for randomness.
            top_p (float, optional): Nucleus sampling parameter.
            top_k (int, optional): Top-k sampling parameter.
            client (httpx.Client): httpx client to use.
            base_url (str): Base URL of the regolo HTTP server.
        Returns:
            dict: Response from the vLLM server.
        """
        if type(messages) == Conversation:
            messages = messages.get_lines() # TODO: to test
        api_key = KeysHandler.check_key(api_key)

        if client is None:
            client = httpx.Client()

        payload = {
            "model": model,
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
            ...
        else:
            response = safe_post(client=client, url=f"{base_url}/v1/chat/completions", json=payload,
                                 headers=headers)

            return response.json()

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
                 full_output: bool = False):
        if user_prompt is not None:
            self.instance.add_prompt_as_role(prompt=user_prompt, role="user")

        if stream is True:
            ...
        else:
            response = self.static_chat_completions(messages=self.instance.get_conversation(),
                                                    model=self.instance.get_model(),
                                                    stream=stream,
                                                    api_key=self.instance.get_api_key(),
                                                    max_tokens=max_tokens,
                                                    temperature=temperature,
                                                    top_p=top_p,
                                                    top_k=top_k,
                                                    client=self.instance.get_client(),
                                                    base_url=self.base_url
                                                    )

            responseRole = response["choices"][0]["message"]["role"]
            responseText = response["choices"][0]["message"]["content"]

            self.instance.add_line(ConversationLine(role=responseRole,
                                                    content=responseText))

            if full_output:
                return response
            else:
                return responseText
