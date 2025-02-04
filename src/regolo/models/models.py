import json

import httpx


class ModelsHandler:
    """
    A utility class for handling model-related operations,
    such as retrieving available models and validating a given model.
    """

    @staticmethod
    def get_models() -> list[str]:
        """
        Retrieves the list of available models from the Regolo server.

        This method fetches the models in JSON format from the Regolo server
        and returns a list of models.

        :return: A list of models (strings).
        """

        # Fetch the models' information from the Regolo server
        models = json.loads(httpx.get("https://regolo.ai/models.json").text)["models"]

        # Return a list of models from the fetched models data
        return [model["id"] for model in models]

    @staticmethod
    def check_model(model: str) -> str:
        """
        Checks if the given model is valid.

        This method checks whether a given model exists in the list of available models.
        If the model is not valid or is None, a RuntimeError is raised.

        :param model: The model ID to be validated.
        :return: The model ID if it is valid.
        :raises RuntimeError: If the model is None or not found in the available models.
        """

        if model is None:
            raise RuntimeError("Model is required")  # Ensure the model is not None
        elif model not in ModelsHandler.get_models():
            raise RuntimeError("Model not found")  # Raise error if the model doesn't exist in the available models

        # TODO: Add handling for a more flexible model request (e.g., fuzzy search or alternatives)
        return model  # Return the model if it's valid
