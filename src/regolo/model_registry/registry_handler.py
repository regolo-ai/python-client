import os
import requests
import json
import logging
from typing import Dict, Optional, Union, List
import time


class GitLabModelRegistryUploader:
    """
    A client for uploading ML models to GitLab Model Registry with Keycloak SSO authentication.
    """

    def __init__(
            self,
            gitlab_url: str,
            keycloak_url: str,
            keycloak_realm: str,
            keycloak_client_id: str,
            keycloak_client_secret: Optional[str] = None,
            verify_ssl: bool = True,
            log_level: int = logging.INFO
    ):
        """
        Initialize the GitLab Model Registry client with Keycloak authentication.

        Args:
            gitlab_url: Base URL of the GitLab instance (e.g., 'https://gitlab.example.com')
            keycloak_url: Base URL of the Keycloak server (e.g., 'https://keycloak.example.com')
            keycloak_realm: Keycloak realm name
            keycloak_client_id: Client ID for the Keycloak client
            keycloak_client_secret: Client secret for the Keycloak client (optional for public clients)
            verify_ssl: Whether to verify SSL certificates
            log_level: Logging level
        """
        self.gitlab_url = gitlab_url.rstrip('/')
        self.keycloak_url = keycloak_url.rstrip('/')
        self.keycloak_realm = keycloak_realm
        self.keycloak_client_id = keycloak_client_id
        self.keycloak_client_secret = keycloak_client_secret
        self.verify_ssl = verify_ssl
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = 0

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("GitLabModelRegistry")

    def authenticate(self, username: Optional[str] = None, password: Optional[str] = None) -> bool:
        """
        Authenticate with Keycloak to obtain access tokens for GitLab API.

        Args:
            username: Keycloak username (for password grant)
            password: Keycloak password (for password grant)

        Returns:
            bool: True if authentication was successful
        """
        token_url = f"{self.keycloak_url}/realms/{self.keycloak_realm}/protocol/openid-connect/token"

        # Determine grant type based on provided credentials
        if username and password:
            # Resource Owner Password Credentials Grant
            data = {
                'grant_type': 'password',
                'client_id': self.keycloak_client_id,
                'username': username,
                'password': password,
                'scope': 'openid'
            }

            # Add client secret if provided
            if self.keycloak_client_secret:
                data['client_secret'] = self.keycloak_client_secret

        elif self.keycloak_client_secret:
            # Client Credentials Grant (for service accounts)
            data = {
                'grant_type': 'client_credentials',
                'client_id': self.keycloak_client_id,
                'client_secret': self.keycloak_client_secret,
                'scope': 'openid'
            }
        else:
            self.logger.error("Authentication failed: Must provide either username/password or client secret")
            return False

        response = requests.post(
            token_url,
            data=data,
            verify=self.verify_ssl
        )

        try:
            response.raise_for_status()

            token_data = response.json()
            self.access_token = token_data['access_token']
            self.refresh_token = token_data.get('refresh_token')

            # Calculate token expiry time
            self.token_expiry = time.time() + token_data.get('expires_in', 300) - 60  # 60s buffer

            self.logger.info("Successfully authenticated with Keycloak")
            return True

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            if response.status_code == 401:
                self.logger.error("Invalid credentials provided")
            elif response.status_code == 400:
                self.logger.error(f"Bad request: {response.text}")
            return False

    def _refresh_token_if_needed(self) -> bool:
        """
        Check if the token is expired and refresh it if needed.

        Returns:
            bool: True if token is valid (refreshed or still valid)
        """
        if time.time() > self.token_expiry and self.refresh_token:
            token_url = f"{self.keycloak_url}/realms/{self.keycloak_realm}/protocol/openid-connect/token"

            data = {
                'grant_type': 'refresh_token',
                'client_id': self.keycloak_client_id,
                'refresh_token': self.refresh_token
            }

            if self.keycloak_client_secret:
                data['client_secret'] = self.keycloak_client_secret

            try:
                response = requests.post(
                    token_url,
                    data=data,
                    verify=self.verify_ssl
                )
                response.raise_for_status()

                token_data = response.json()
                self.access_token = token_data['access_token']
                self.refresh_token = token_data.get('refresh_token')

                # Calculate token expiry time
                self.token_expiry = time.time() + token_data.get('expires_in', 300) - 60  # 60s buffer

                self.logger.info("Successfully refreshed the token")
                return True

            except requests.exceptions.RequestException as e:
                self.logger.error(f"Token refresh failed: {str(e)}")
                return False

        return self.access_token is not None

    def _make_api_request(
            self,
            method: str,
            endpoint: str,
            data: Optional[Dict] = None,
            files: Optional[Dict] = None,
            json_data: Optional[Dict] = None,
            params: Optional[Dict] = None
    ) -> Dict:
        """
        Make an authenticated request to the GitLab API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (will be appended to the GitLab API URL)
            data: Form data to send
            files: Files to upload
            json_data: JSON data to send in the request body
            params: URL parameters

        Returns:
            Dict: Response data

        Raises:
            Exception: If the request fails
        """
        if not self._refresh_token_if_needed():
            raise Exception("No valid authentication token available")

        url = f"{self.gitlab_url}/api/v4/{endpoint.lstrip('/')}"
        headers = {
            'Authorization': f'Bearer {self.access_token}'
        }

        response = requests.request(
            method,
            url,
            headers=headers,
            data=data,
            files=files,
            json=json_data,
            params=params,
            verify=self.verify_ssl
        )

        try:

            response.raise_for_status()

            # Return empty dict for 204 No Content
            if response.status_code == 204:
                return {}

            return response.json()

        except requests.exceptions.HTTPError as e:
            self.logger.error(f"API request failed: {str(e)}")
            if response.status_code == 401:
                self.logger.error("Authentication token might be invalid or expired")

            error_msg = f"GitLab API error: {response.status_code}"
            try:
                error_data = response.json()
                if "message" in error_data:
                    error_msg += f" - {error_data['message']}"
                elif "error" in error_data:
                    error_msg += f" - {error_data['error']}"
            except:
                error_msg += f" - {response.text}"

            raise Exception(error_msg)

    def list_projects(self, search: Optional[str] = None) -> dict:
        """
        List GitLab projects the user has access to.

        Args:
            search: Optional search query

        Returns:
            List[Dict]: List of projects
        """
        params: dict = {'membership': True}

        if search:
            params['search'] = search

        return self._make_api_request('GET', 'projects', params=params)

    def upload_model(
            self,
            project_id: Union[int, str],
            model_name: str,
            model_file_path: str,
            version: str,
            description: Optional[str] = None,
            metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Upload a model to the GitLab Model Registry.

        Args:
            project_id: Project ID or path (e.g. 'group/project')
            model_name: Name of the model
            model_file_path: Path to the model file
            version: Version of the model
            description: Description of the model version
            metadata: Additional metadata for the model

        Returns:
            Dict: Response data with information about the uploaded model
        """
        self.logger.info(f"Uploading model {model_name} version {version} to project {project_id}")

        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")

        # First, create or get the model
        try:
            model = self._make_api_request('GET', f'projects/{project_id}/ml/models/{model_name}')
            self.logger.info(f"Model {model_name} already exists, id: {model['id']}")
        except Exception:
            # Model doesn't exist, create it
            self.logger.info(f"Creating new model {model_name} in project {project_id}")
            model = self._make_api_request(
                'POST',
                f'projects/{project_id}/ml/models',
                json_data={
                    'name': model_name,
                    'description': description or f"Model {model_name}"
                }
            )

        model_id = model['id']

        # Create model version
        version_data = {
            'version': version
        }

        if description:
            version_data['description'] = description

        if metadata:
            version_data['metadata'] = json.dumps(metadata)

        model_version = self._make_api_request(
            'POST',
            f'projects/{project_id}/ml/models/{model_id}/versions',
            json_data=version_data
        )

        version_id = model_version['id']

        # Upload the model file
        with open(model_file_path, 'rb') as f:
            files = {'file': f}

            response = self._make_api_request(
                'POST',
                f'projects/{project_id}/ml/models/{model_id}/versions/{version_id}/files',
                files=files
            )

        self.logger.info(f"Successfully uploaded model {model_name} version {version}")
        return model_version

    def list_model_versions(self, project_id: Union[int, str], model_name: str) -> dict:
        """
        List all versions of a model.

        Args:
            project_id: Project ID or path
            model_name: Name of the model

        Returns:
            List[Dict]: List of model versions
        """
        try:
            model = self._make_api_request('GET', f'projects/{project_id}/ml/models/{model_name}')
            model_id = model['id']

            return self._make_api_request('GET', f'projects/{project_id}/ml/models/{model_id}/versions')
        except Exception as e:
            self.logger.error(f"Failed to list model versions: {str(e)}")
            raise

    def download_model(
            self,
            project_id: Union[int, str],
            model_name: str,
            version: str,
            output_path: str
    ) -> str:
        """
        Download a specific model version.

        Args:
            project_id: Project ID or path
            model_name: Name of the model
            version: Version of the model to download
            output_path: Directory path to save the downloaded model

        Returns:
            str: Path to the downloaded file
        """
        try:
            # Get model ID
            model = self._make_api_request('GET', f'projects/{project_id}/ml/models/{model_name}')
            model_id = model['id']

            # Get version ID
            versions = self._make_api_request('GET', f'projects/{project_id}/ml/models/{model_id}/versions')
            version_id = None

            for v in versions:
                if v['version'] == version:
                    version_id = v['id']
                    break

            if not version_id:
                raise Exception(f"Version {version} not found for model {model_name}")

            # Get file info
            files = self._make_api_request(
                'GET',
                f'projects/{project_id}/ml/models/{model_id}/versions/{version_id}/files'
            )

            if not files:
                raise Exception(f"No files found for model {model_name} version {version}")

            # Ensure output directory exists
            os.makedirs(output_path, exist_ok=True)

            # Download each file
            downloaded_files = []
            for file_info in files:
                file_id = file_info['id']
                filename = file_info['filename']

                # Download file
                if not self._refresh_token_if_needed():
                    raise Exception("No valid authentication token available")

                url = f"{self.gitlab_url}/api/v4/projects/{project_id}/ml/models/{model_id}/versions/{version_id}/files/{file_id}/download"
                headers = {
                    'Authorization': f'Bearer {self.access_token}'
                }

                response = requests.get(url, headers=headers, verify=self.verify_ssl, stream=True)
                response.raise_for_status()

                file_path = os.path.join(output_path, filename)
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                downloaded_files.append(file_path)
                self.logger.info(f"Downloaded {filename} to {file_path}")

            return downloaded_files[0] if len(downloaded_files) == 1 else downloaded_files

        except Exception as e:
            self.logger.error(f"Failed to download model: {str(e)}")
            raise


# Example

if __name__ == "__main__":

    gitlab_url = "https://gitlab.example.com"
    keycloak_url = "https://keycloak.example.com"
    keycloak_realm = "my-realm"
    keycloak_client_id = "gitlab-model-registry-client"
    keycloak_client_secret = "your-client-secret"

    client = GitLabModelRegistryUploader(
        gitlab_url=gitlab_url,
        keycloak_url=keycloak_url,
        keycloak_realm=keycloak_realm,
        keycloak_client_id=keycloak_client_id,
        keycloak_client_secret=keycloak_client_secret
    )

    # Authenticate with username/password
    client.authenticate(username="user@example.com", password="password")

    # Or authenticate with client credentials (service account)
    # client.authenticate()

    # Upload a model
    client.upload_model(
        project_id="my-group/my-project",
        model_name="awesome-model",
        model_file_path="/path/to/model.pkl",
        version="1.0.0",
        description="Initial release of awesome model",
        metadata={
            "framework": "scikit-learn",
            "accuracy": 0.95,
            "training_date": "2025-04-01"
        }
    )

    # List model versions
    versions = client.list_model_versions(
        project_id="my-group/my-project",
        model_name="awesome-model"
    )
    print(f"Available versions: {versions}")

    # Download a model
    downloaded_path = client.download_model(
        project_id="my-group/my-project",
        model_name="awesome-model",
        version="1.0.0",
        output_path="/path/to/download/directory"
    )
    print(f"Downloaded model to {downloaded_path}")