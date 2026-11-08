from typing import Any, Literal, overload

import httpx

import regolo
from regolo.keys.keys import KeysHandler
from regolo.transport import api_url, json_object, managed_http


class ModelsHandler:
    """
    A utility class for handling model-related operations,
    such as retrieving available models and validating a given model.
    """

    @staticmethod
    @overload
    def get_models(
        base_url: str,
        api_key: str,
        model_info: Literal[False] = False,
        client: httpx.Client | None = None,
    ) -> list[str]: ...

    @staticmethod
    @overload
    def get_models(
        base_url: str,
        api_key: str,
        model_info: Literal[True],
        client: httpx.Client | None = None,
    ) -> list[dict[str, Any]]: ...

    @staticmethod
    @overload
    def get_models(
        base_url: str,
        api_key: str,
        model_info: bool,
        client: httpx.Client | None = None,
    ) -> list[str] | list[dict[str, Any]]: ...

    @staticmethod
    @managed_http
    def get_models(
        base_url: str, api_key: str, model_info: bool = False, client: httpx.Client | None = None
    ) -> list[str] | list[dict]:
        """
        Retrieves the list of available models from the Regolo server.

        This method fetches the models in JSON format from the Regolo server
        and returns a list of models.

        :param base_url: The Regolo server base URL.
        :param api_key: The API key for the Regolo server authentication.
        :param model_info: Whether to retrieve information about the model (Defaults to False)
        :param client: Caller-owned HTTPX transport, or None for a temporary transport.

        :return: A list of models (strings).
        """
        assert client is not None  # Supplied by managed_http when omitted by the caller.
        headers = {"Authorization": KeysHandler.check_key(api_key)}

        path = "/model/info" if model_info else "/models"
        data = json_object(client.get(api_url(base_url, path), headers=headers)).get("data")
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError("Model discovery response must contain a data array of objects")
        if model_info:
            return data
        if any(not isinstance(row.get("id"), str) or not row["id"] for row in data):
            raise ValueError("Model discovery contains a missing or invalid model ID")
        return [row["id"] for row in data]

    @staticmethod
    def check_model(
        model: str | None,
        base_url: str | None,
        api_key: str,
        client: httpx.Client | None = None,
        *,
        enabled: bool | None = None,
    ) -> str | None:
        """
        Checks if the given model is valid.

        This method checks whether a given model exists in the list of available models.
        If the model is not valid or is None, a RuntimeError is raised.

        :param model: The model ID to be validated.
        :param base_url: The base URL of the Regolo server.
        :param api_key: The API key of the Regolo server.
        :param client: Caller-owned HTTPX transport used for the availability check.

        :return: The model ID if it is valid.
        :raises RuntimeError: If the model is None or not found in the available models.
        """
        if not isinstance(model, str) or not model.strip():
            raise RuntimeError("Model is required")  # Ensure the model is not None
        if not (regolo.enable_model_checks if enabled is None else enabled):
            return model
        if model not in ModelsHandler.get_models(
            base_url=base_url or "https://api.regolo.ai", api_key=api_key, client=client
        ):
            raise RuntimeError("Model not found")

        return model  # Return the model if it's valid

    @staticmethod
    @managed_http
    def get_catalog(
        base_url: str, api_key: str, client: httpx.Client | None = None
    ) -> list[dict[str, Any]]:
        """Full user model catalog: capabilities, limits, prices and future fields."""
        assert client is not None
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        response = client.get(
            api_url(base_url, "/model_group/info"),
            headers={"Authorization": KeysHandler.check_key(api_key)},
        )
        data = json_object(response).get("data")
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError("Model catalog response must contain a data array of objects")
        return data
