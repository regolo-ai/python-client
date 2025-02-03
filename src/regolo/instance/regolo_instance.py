from typing import Optional, List, Dict

import httpx

from regolo.instance.structures.conversation_model import Conversation, ConversationLine

from regolo.models.models import ModelsHandler
from regolo.keys.keys import KeysHandler


class RegoloInstance:
    def __init__(self, model: str, api_key: str, client: Optional[httpx.Client],
                 previous_conversations: Optional[Conversation] = None) -> None:
        self.conversation: Conversation = Conversation(
            lines=[]) if previous_conversations is None else previous_conversations
        self.client: httpx.Client = httpx.Client() if client is None else client
        self.api_key = KeysHandler.check_key(api_key)
        self.model: str = model

    def get_client(self) -> httpx.Client:
        return self.client

    def get_api_key(self) -> str:
        return self.api_key

    def get_model(self) -> str:
        return self.model

    def change_model(self, new_model: str) -> None:
        self.model = ModelsHandler.check_model(new_model)

    def get_conversation(self) -> List[Dict[str, str]]:
        """Get conversation as list of dicts"""
        return self.conversation.get_lines()

    def add_line(self, conversation: ConversationLine) -> None:
        """Adds a line to the conversation"""
        self.conversation.lines.append(conversation)

    def overwrite_conversation(self, conversation: Conversation) -> None:
        """Overwrites the conversation with the given one"""
        self.conversation = conversation

    def clear_conversation(self) -> None:
        """Clears the conversation to start a new chat from scratch"""
        self.conversation = Conversation(lines=[])

    def add_prompt_as_role(self, prompt: str, role: str) -> None:
        """Adds a prompt as given role to the conversation.
        Normally roles are "user" and "assistant", but they can vary based on model."""
        self.conversation.lines.append(ConversationLine(role=role, content=prompt))
