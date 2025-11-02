from typing import Dict, List, Optional

import httpx

from regolo.instance.structures.conversation_model import Conversation, ConversationLine
from regolo.keys.keys import KeysHandler
from regolo.models.models import ModelsHandler


class RegoloInstance:
    """
    Represents an instance of a Regolo AI client, maintaining the conversation state,
    HTTP client, API key, and selected model.

    :param chat_model: The Regolo AI model to use.
    :param api_key: The API key for authentication.
    :param client: An optional existing `httpx.Client` instance for making requests.
    :param previous_conversations: An optional `Conversation` instance to maintain chat history.
    """

    def __init__(self,
                 chat_model: Optional[str],
                 embedder_model: Optional[str],
                 image_generation_model: Optional[str],
                 audio_transcription_model: Optional[str],
                 reranker_model: Optional[str],
                 api_key: str,
                 base_url: str,
                 client: Optional[httpx.Client],
                 previous_conversations: Optional[Conversation] = None) -> None:
        """
        Initializes a RegoloInstance.

        :param chat_model: The selected AI model to use.
        :param api_key: The API key for authentication.
        :param client: Optional `httpx.Client` instance; if not provided, a new one is created.
        :param previous_conversations: Optional `Conversation` instance for maintaining chat history.
        """
        self.conversation: Conversation = Conversation(
            lines=[]) if previous_conversations is None else previous_conversations
        self.client: httpx.Client = httpx.Client() if client is None else client
        self.api_key: str = KeysHandler.check_key(api_key)
        self.chat_model: Optional[str] = chat_model
        self.image_generation_model: Optional[str] = image_generation_model
        self.audio_transcription_model: Optional[str] = audio_transcription_model
        self.embedder_model: Optional[str] = embedder_model
        self.reranker_model: Optional[str] = reranker_model
        self.base_url: str = base_url

    def get_client(self) -> httpx.Client:
        """
        Returns the `httpx.Client` instance used for making requests.

        :return: The `httpx.Client` instance.
        """
        return self.client

    def get_api_key(self) -> str:
        """
        Returns the stored API key.

        :return: The API key as a string.
        """
        return self.api_key

    def get_model(self) -> str:
        """
        Returns the selected AI model.

        :return: The model name as a string.
        """
        return self.chat_model

    def get_base_url(self) -> str:
        """
        Returns the base url.

        :return: The base url.
        """
        return self.base_url

    def get_image_model(self) -> str:
        """
        Returns the image model.

        :return: The image name as a string.
        """
        return self.image_generation_model

    def get_embedder_model(self) -> str:
        """
        Returns the embedder model.

        :return: The embedder model as a string.
        """
        return self.embedder_model

    def get_reranker_model(self) -> Optional[str]:
        """
        Returns the reranker model.

        :return: The reranker model as a string
        """
        return self.reranker_model

    def change_model(self, new_model: str) -> None:
        """
        Changes the model of this instance to a new one.

        :param new_model: The new model name to switch to.
        """
        self.chat_model = ModelsHandler.check_model(model=new_model, base_url=self.base_url, api_key=self.api_key)

    def get_conversation(self) -> List[Dict[str, str]]:
        """
        Gets conversation as a list of dictionaries.

        :return: A list of dictionaries representing the conversation history.
        """
        return self.conversation.get_lines()

    def add_line(self, conversation: ConversationLine) -> None:
        """
        Adds a new line to the conversation.

        :param conversation: A `ConversationLine` instance containing the role and message content.
        """
        self.conversation.lines.append(conversation)

    def overwrite_conversation(self, conversation: Conversation) -> None:
        """
        Replaces the existing conversation with a new one

        :param conversation: A `Conversation` instance containing the new conversation history.
        """
        self.conversation = conversation

    def clear_conversation(self) -> None:
        """Clears the conversation history, resetting it to an empty state."""
        self.conversation = Conversation(lines=[])

    def add_prompt_as_role(self, prompt: str, role: str) -> None:
        """
        Adds a prompt to the conversation under a specific role.
        Normally roles are "user" and "assistant", but they can vary with the model used.

        :param prompt: The message content to add.
        :param role: The role of the speaker (e.g., "user", "assistant").
        """
        self.conversation.lines.append(ConversationLine(role=role, content=prompt))
