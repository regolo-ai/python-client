import json

from pydantic import BaseModel
from typing import List, Dict

from pydantic import StrictStr


class ConversationLine(BaseModel):
    role: StrictStr
    content: StrictStr


class Conversation(BaseModel):
    lines: List[ConversationLine]

    def __init__(self, lines: List[ConversationLine]):
        super().__init__(lines=lines)

    def to_json(self):
        return json.dumps(self.model_dump(), sort_keys=True)

    def get_lines(self) -> List[Dict[str, str]]:
        return json.loads((self.to_json()))["lines"]

    def print_conversation(self):
        for line in self.get_lines():
            print(f"{line["role"]}: {line["content"]}\n")
