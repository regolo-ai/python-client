import httpx

import regolo


class ModelsHandler:
    """
    A utility class for handling model-related operations,
    such as retrieving available models and validating a given model.
    """

    @staticmethod
    def get_models(base_url: str, api_key: str, model_info: bool=False) -> list[str] | list[dict]:
        """
        Retrieves the list of available models from the Regolo server.

        This method fetches the models in JSON format from the Regolo server
        and returns a list of models.

        :param base_url: The Regolo server base URL.
        :param api_key: The API key for the Regolo server authentication.
        :param model_info: Whether to retrieve information about the model (Defaults to False)

        :return: A list of models (strings).
        """
        headers = {"Authorization": f"{api_key}"}

        # Fetch the models' information from the Regolo server
        if model_info:
            response = httpx.get(f"{base_url}/model/info", headers=headers)

            if response.status_code == 401:
                raise Exception("Authentication failed. Couldn't fetch models")
            if response.status_code != 200:
                raise Exception("Failed to fetch models")
            models_info = response.json()["data"]

            # Return a list of models from the fetched models data
            return models_info

        else:
            response = httpx.get(f"{base_url}/models", headers=headers)

            if response.status_code == 401:
                raise Exception("Authentication failed. Couldn't fetch models")
            if response.status_code != 200:
                raise Exception("Failed to fetch models")

            models_info = response.json()["data"]

            # Return a list of models from the fetched models data
            return [model["id"] for model in models_info]

    @staticmethod
    def check_model(model: str, base_url: str, api_key: str) -> str:
        """
        Checks if the given model is valid.

        This method checks whether a given model exists in the list of available models.
        If the model is not valid or is None, a RuntimeError is raised.

        :param model: The model ID to be validated.
        :param base_url: The base URL of the Regolo server.
        :param api_key: The API key of the Regolo server.

        :return: The model ID if it is valid.
        :raises RuntimeError: If the model is None or not found in the available models.
        """
        if not regolo.enable_model_checks:
            return model

        if model is None:
            raise RuntimeError("Model is required")  # Ensure the model is not None
        elif model not in ModelsHandler.get_models(base_url=base_url, api_key=api_key):
            raise RuntimeError("Model not found")

        # TODO: Add handling for a more flexible model request (e.g., fuzzy search or alternatives)
        return model  # Return the model if it's valid
