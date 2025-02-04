from typing import Dict
from typing import List
from typing import Optional

import httpx

from regolo.instance.structures.conversation_model import Conversation
from regolo.instance.structures.conversation_model import ConversationLine
from regolo.keys.keys import KeysHandler
from regolo.models.models import ModelsHandler


class RegoloInstance:
    """
    Represents an instance of a Regolo AI client, maintaining the conversation state,
    HTTP client, API key, and selected model.

    :param model: The Regolo AI model to use.
    :param api_key: The API key for authentication.
    :param client: An optional existing `httpx.Client` instance for making requests.
    :param previous_conversations: An optional `Conversation` instance to maintain chat history.
    """

    def __init__(self, model: str, api_key: str, client: Optional[httpx.Client],
                 previous_conversations: Optional[Conversation] = None) -> None:
        """
        Initializes a RegoloInstance.

        :param model: The selected AI model to use.
        :param api_key: The API key for authentication.
        :param client: Optional `httpx.Client` instance; if not provided, a new one is created.
        :param previous_conversations: Optional `Conversation` instance for maintaining chat history.
        """
        self.conversation: Conversation = Conversation(
            lines=[]) if previous_conversations is None else previous_conversations
        self.client: httpx.Client = httpx.Client() if client is None else client
        self.api_key = KeysHandler.check_key(api_key)
        self.model: str = model

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
        return self.model

    def change_model(self, new_model: str) -> None:
        """
        Changes the model of this instance to a new one.

        :param new_model: The new model name to switch to.
        """
        self.model = ModelsHandler.check_model(new_model)

    def get_conversation(self) -> List[Dict[str, str]]:
        """
        Gets conversation as list of dictionaries.

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
        Normally roles are "user" and "assistant", but they can vary based on model.

        :param prompt: The message content to add.
        :param role: The role of the speaker (e.g., "user", "assistant").
        """
        self.conversation.lines.append(ConversationLine(role=role, content=prompt))
