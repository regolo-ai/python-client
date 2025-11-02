import json
from typing import Dict, List

from pydantic import BaseModel, StrictStr


class ConversationLine(BaseModel):
    """
    Represents a single message in a conversation.

    :param role: The role of the message sender (e.g., "user" or "assistant").
    :param content: The actual text content of the message.
    """

    role: StrictStr
    content: StrictStr


class Conversation(BaseModel):
    """
    Represents a conversation consisting of multiple lines.

    :param lines: A list of ConversationLine objects representing the conversation history.
    """

    lines: List[ConversationLine]

    def __init__(self, lines: List[ConversationLine]):
        """
        Initializes a Conversation instance.

        :param lines: A list of ConversationLine instances representing the dialogue.
        """
        super().__init__(lines=lines)

    def to_json(self):
        """
        Converts the conversation object to a JSON string.

        :return: A JSON string representation of the conversation.
        """
        return json.dumps(self.model_dump(), sort_keys=True)

    def get_lines(self) -> List[Dict[str, str]]:
        """
        Retrieves the conversation lines as a list of dictionaries.

        :return: A list of dictionaries with "role" and "content" keys.
        """
        return json.loads((self.to_json()))["lines"]

    def print_conversation(self):
        """Prints the conversation in a readable format, with each message prefixed by the role."""
        for line in self.get_lines():
            print(f'{line["role"]}: {line["content"]}\n')
