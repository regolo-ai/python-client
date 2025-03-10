class KeysHandler:
    """A utility class for handling API keys, ensuring they follow the correct format."""

    @staticmethod
    def fix_key(api_key: str) -> str:
        """
        Ensures the API key has the correct 'Bearer' prefix.

        :param api_key: The API key to be formatted.
        :return: A properly formatted API key with the 'Bearer' prefix.
        """
        if not api_key.startswith("Bearer "):
            return "Bearer " + api_key
        return api_key

    @staticmethod
    def check_key(api_key: str) -> str:
        """
        Validates and formats the API key.

        :param api_key: The API key to be validated.

        :return: A properly formatted API key.
        :raises RuntimeError: If the API key is None.
        """
        if api_key is None:
            raise RuntimeError("API key is required")
        return KeysHandler.fix_key(api_key)
