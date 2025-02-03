class KeysHandler:
    @staticmethod
    def fix_key(api_key: str) -> str:
        if not api_key.startswith("Bearer "):
            return "Bearer " + api_key
        return api_key

    @staticmethod
    def check_key(api_key: str) -> str:
        if api_key is None:
            raise RuntimeError("API key is required")
        return KeysHandler.fix_key(api_key)
