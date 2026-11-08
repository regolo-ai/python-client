import json
import os
import tempfile
from pathlib import Path
from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field, StrictStr


class ConversationLine(BaseModel):
    """
    Represents a single message in a conversation.

    :param role: The role of the message sender (e.g., "user" or "assistant").
    :param content: The actual text content of the message.
    """

    model_config = ConfigDict(extra="allow")
    role: StrictStr
    content: StrictStr | list[dict[str, Any]] | None = None
    tool_call_id: StrictStr | None = None
    tool_calls: list[dict[str, Any]] | None = None
    name: StrictStr | None = None


class Conversation(BaseModel):
    """
    Represents a conversation consisting of multiple lines.

    :param lines: A list of ConversationLine objects representing the conversation history.
    """

    lines: List[ConversationLine] = Field(default_factory=list)

    @classmethod
    def from_messages(cls, messages: list[dict[str, Any]]) -> "Conversation":
        return cls(lines=[ConversationLine.model_validate(message) for message in messages])

    def append(
        self, role: str, content: str | list[dict[str, Any]] | None = None, **fields: Any
    ) -> "Conversation":
        self.lines.append(ConversationLine(role=role, content=content, **fields))
        return self

    def fork(self) -> "Conversation":
        """Independent history, including nested multimodal and tool content."""
        return self.model_copy(deep=True)

    def trim(self, max_turns: int, *, keep_system: bool = True) -> None:
        """Keep complete user turns, including their assistant/tool exchanges.

        This is a message-window helper, not a token estimator. Initial
        system/developer instructions are retained by default.
        """
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        starts = [i for i, line in enumerate(self.lines) if line.role == "user"]
        if len(starts) <= max_turns:
            return
        prefix = []
        if keep_system:
            for line in self.lines:
                if line.role not in {"system", "developer"}:
                    break
                prefix.append(line)
        self.lines = prefix + self.lines[starts[-max_turns] :]

    def save(self, destination: str | Path) -> None:
        """Atomic owner-only message JSON, without serializing client credentials."""
        path = Path(destination).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".regolo-chat-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"lines": self.get_lines()}, ensure_ascii=False))
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, source: str | Path) -> "Conversation":
        return cls.model_validate_json(Path(source).expanduser().read_text(encoding="utf-8"))

    def to_json(self):
        """
        Converts the conversation object to a JSON string.

        :return: A JSON string representation of the conversation.
        """
        return json.dumps(self.model_dump(), sort_keys=True)

    def get_lines(self) -> list[dict[str, Any]]:
        """
        Retrieves the conversation lines as a list of dictionaries.

        :return: A list of dictionaries with "role" and "content" keys.
        """
        return [line.model_dump(exclude_unset=True) for line in self.lines]

    def print_conversation(self):
        """Prints the conversation in a readable format, with each message prefixed by the role."""
        for line in self.get_lines():
            print(f"{line['role']}: {line.get('content')}\n")
