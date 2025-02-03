import json

import httpx


class ModelsHandler:
    @staticmethod
    def get_models() -> list[str]:
        models = json.loads(httpx.get("https://regolo.ai/models.json").text)["models"]
        return [model["id"] for model in models]

    @staticmethod
    def check_model(model: str) -> str:
        if model is None:
            raise RuntimeError("Model is required")
        elif model not in ModelsHandler.get_models():
            raise RuntimeError("Model not found")
        # TODO: fix model request if requested model is understandable
        return model
