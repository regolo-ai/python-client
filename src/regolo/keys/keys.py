class KeysHandler:
    """A utility class for handling API keys, ensuring they follow the correct format."""

    @staticmethod
    def fix_key(api_key: str) -> str:
        """
        Ensures the API key has the correct 'Bearer' prefix.

        :param api_key: The API key to be formatted.
        :return: A properly formatted API key with the 'Bearer' prefix.
        """
        api_key = api_key.strip()
        if api_key.lower().startswith("bearer "):
            api_key = api_key.split(" ", 1)[1].strip()
        if not api_key or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in api_key):
            raise RuntimeError("API key must be a nonempty token without whitespace")
        return "Bearer " + api_key

    @staticmethod
    def check_key(api_key: str | None) -> str:
        """
        Validates and formats the API key.

        :param api_key: The API key that has to be validated.

        :return: A properly formatted API key.
        :raises RuntimeError: If the API key is None.
        """
        if (
            not isinstance(api_key, str)
            or not api_key.strip()
            or api_key.strip().lower() == "bearer"
        ):
            raise RuntimeError("API key is required")
        return KeysHandler.fix_key(api_key.strip())
