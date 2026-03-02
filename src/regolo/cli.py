import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import json
import pprint
import re
from datetime import datetime
from io import BytesIO
from typing import Any, Optional

import click
import httpx
from PIL import Image

import regolo
from regolo import RegoloClient
from regolo.keys.keys import KeysHandler

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
AUDIO_EXTENSIONS = ["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"]

# Configuration file for storing auth tokens
CONFIG_FILE = os.path.expanduser("~/.regolo_config.json")


class ModelManagementClient:
    """Client for interacting with the Model Management API"""

    def __init__(self, base_url: str = "https://devmid.regolo.ai"):
        self.base_url = base_url
        self.token = None
        self.refresh_token = None
        self.timeout = 60  # timeout in seconds
        self._load_config()

    def _load_config(self):
        """Load authentication tokens from config file"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.token = config.get('access_token')
                    self.refresh_token = config.get('refresh_token')
            except Exception as e:
                click.echo(f"Failed to load config: {e}")

    def _save_config(self):
        """Save authentication tokens to config file"""
        config = {
            'access_token': self.token,
            'refresh_token': self.refresh_token
        }
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)

    def _headers(self):
        """Get authentication headers"""
        if not self.token:
            raise Exception("Not authenticated. Please run 'regolo auth login' first.")
        return {"Authorization": f"Bearer {self.token}"}

    def _make_request(self, method: str, endpoint: str, **kwargs):
        """Make authenticated API request with token refresh handling"""
        url = f"{self.base_url}{endpoint}"
        response = None
        try:
            response = httpx.request(method, url, headers=self._headers(), timeout=self.timeout, **kwargs)

            # If the token expired, try to refresh
            if response.status_code == 401 and self.refresh_token:
                if self._refresh_token():
                    # Retry with new token
                    response = httpx.request(method, url, headers=self._headers(), timeout=self.timeout, **kwargs)
                else:
                    raise Exception("Authentication failed. Please login again.")

            response.raise_for_status()
            return response.json() if response.content else {}

        except httpx.HTTPError as e:

            try:
                error_detail = e.request.__dict__.get('detail', str(e))
            except (Exception,):
                error_detail = str(e)
            raise Exception(f"API Error: {error_detail}. Request error: {response.json()}")

    def authenticate(self, username: str, password: str):
        """Authenticate and get access tokens"""
        response = httpx.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password}
        )

        if response.status_code != 200:
            try:
                error = response.json().get('detail', 'Authentication failed')
            except (Exception,):
                error = 'Authentication failed'
            raise Exception(error)

        data = response.json()
        self.token = data["access_token"]
        self.refresh_token = data["refresh_token"]
        self._save_config()
        return data

    def _refresh_token(self):
        """Refresh access token"""
        if not self.refresh_token:
            return False

        try:
            response = httpx.post(
                f"{self.base_url}/auth/refresh",
                json={"refresh_token": self.refresh_token}
            )

            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                self.refresh_token = data["refresh_token"]
                self._save_config()
                return True
        except (Exception,):
            pass

        return False

    # Model Management Methods
    def register_model(self, name: str, provider: str,
                       url: Optional[str] = None, api_key: Optional[str] = None,
                       force: bool = False):
        """Register a new model"""
        data = {
            "name": name,
            "provider": provider,
            "force": force
        }
        if url:
            data["url"] = url
        if api_key:
            data["api_key"] = api_key

        return self._make_request("POST", "/models/load", json=data)

    def get_models(self):
        """Get all models"""
        return self._make_request("GET", "/models")

    def get_model(self, model_name: str):
        """Get specific model details"""
        return self._make_request("GET", f"/models/{model_name}")

    def delete_model(self, model_name: str):
        """Delete a model"""
        return self._make_request("DELETE", f"/models/{model_name}")

    # SSH Key Management Methods
    def add_ssh_key(self, title: str, key: str):
        """Add SSH key"""
        return self._make_request("POST", "/ssh-keys", json={"title": title, "key": key})

    def get_ssh_keys(self):
        """Get all SSH keys"""
        return self._make_request("GET", "/ssh-keys")

    def delete_ssh_key(self, key_id: str):
        """Delete SSH key"""
        return self._make_request("DELETE", f"/ssh-keys/{key_id}")

    # Inference Management Methods
    def get_available_gpus(self):
        """Get available GPUs"""
        return self._make_request("GET", "/inference/gpus")

    def load_model_for_inference(self, model_name: str, gpu: str, force: bool = False,
                                 vllm_config: Optional[dict] = None):
        """Load model for inference"""
        data: dict[str, Any]= {"model_name": model_name, "gpu": gpu, "force": force}

        if vllm_config:
            data["vllm_config"] = vllm_config

        return self._make_request("POST", "/inference/load", json=data)

    def unload_model_from_inference(self, session_id: int):
        """Unload model from inference"""
        return self._make_request("POST", "/inference/unload",
                                  json={"session_id": session_id})

    def get_loaded_models(self):
        """Get currently loaded models"""
        return self._make_request("GET", "/inference/loaded-models")

    def get_user_inference_status(self, month: Optional[str] = None, time_range_start: Optional[str] = None,
                                  time_range_end: Optional[str] = None):
        """Get inference status for user's models"""
        params = {}
        if month:
            params['month'] = month
        if time_range_start:
            params['time_range_start'] = time_range_start
        if time_range_end:
            params['time_range_end'] = time_range_end
        return self._make_request("GET", "/inference/user-status", params=params)


# Initialize global client
model_client = ModelManagementClient()


@click.group()
def cli():
    pass


# Authentication Commands
@click.group()
def auth():
    """Authentication commands"""
    pass


@auth.command("login")
@click.option('--username', prompt=True, help='Username for authentication')
@click.option('--password', prompt=True, hide_input=True, help='Password for authentication')
def login(username: str, password: str):
    """Login and save authentication tokens"""
    try:
        result = model_client.authenticate(username, password)
        click.echo(f"✅ Successfully authenticated! Token expires in {result.get('expires_in', 'unknown')} seconds.")
    except Exception as e:
        click.echo(f"❌ Authentication failed: {e}")
        exit(1)


@auth.command("logout")
def logout():
    """Logout and clear saved tokens"""
    if os.path.exists(CONFIG_FILE):
        os.remove(CONFIG_FILE)
    click.echo("✅ Successfully logged out!")


# Model Management Commands
@click.group()
def models():
    """Model management commands"""
    pass


@models.command("register")
@click.option('--name', required=True, help='Name for the model')
@click.option('--type', 'model_type', type=click.Choice(['huggingface', 'ollama', 'custom']),
              required=True, help='Type of model (huggingface or custom)')
@click.option('--url', help='HuggingFace URL (required for huggingface models)')
@click.option('--api-key', help='HuggingFace API key (optional, for private models)')
def register_model(name: str, model_type: str, url: Optional[str],
                   api_key: Optional[str]):
    """Register a new model in the system"""
    try:

        if model_type=="huggingface" and not url:
            click.echo("❌ URL is required for HuggingFace models")
            exit(1)

        model_client.register_model(
            name=name,
            provider=model_type,
            url=url,
            api_key=api_key
        )

        click.echo(f"✅ Model '{name}' registered successfully!")
        if model_type=="huggingface":
            click.echo("📥 HuggingFace model added to your regolo account! (remember to download it if you want to preserve it, since we do not store huggingface models locally)")
        else:
            click.echo("📂 Custom project created. You can now upload your model files using SSH")

    except Exception as e:
        click.echo(f"❌ Failed to register model: {e}")
        exit(1)


@models.command("list")
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def list_models(output_format: str):
    """List all registered models"""
    try:
        result = model_client.get_models()
        models_list = result.get('models', [])

        if not models_list:
            click.echo("No models found")
            return

        if output_format == 'json':
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📋 Found {result.get('total', 0)} models:\n")
            for model in models_list:
                model_type = model['provider']
                click.echo(f"  • {model['name']} ({model_type})")
                if model.get('url'):
                    click.echo(f"    URL: {model['url']}")
                click.echo(f"    Created: {model['created_at']}")
                click.echo()

    except Exception as e:
        click.echo(f"❌ Failed to list models: {e}")
        exit(1)


@models.command("details")
@click.argument('model_name')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def model_details(model_name: str, output_format: str):
    """Get detailed information about a specific model"""
    try:
        model = model_client.get_model(model_name)

        if output_format == 'json':
            click.echo(json.dumps(model, indent=2))
        else:
            model_type = model['provider']
            click.echo(f"\n📋 Model Details: {model['name']}")
            click.echo(f"  Type: {model_type}")
            click.echo(f"  Email: {model['email']}")
            if model.get('url'):
                click.echo(f"  URL: {model['url']}")
            click.echo(f"  Created: {model['created_at']}")

    except Exception as e:
        click.echo(f"❌ Failed to get model details: {e}")
        exit(1)


@models.command("delete")
@click.argument('model_name')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def delete_model(model_name: str, confirm: bool):
    """Delete a model"""
    try:
        if not confirm:
            if not click.confirm(f"Are you sure you want to delete model '{model_name}'?"):
                click.echo("Deletion cancelled")
                return

        model_client.delete_model(model_name)
        click.echo(f"✅ Model '{model_name}' deleted successfully!")

    except Exception as e:
        click.echo(f"❌ Failed to delete model: {e}")
        exit(1)


# SSH Key Management Commands
@click.group()
def ssh():
    """SSH key management commands"""
    pass


@ssh.command("add")
@click.option('--title', required=True, help='Title for the SSH key')
@click.option('--key-file', help='Path to SSH public key file')
@click.option('--key', help='SSH public key content (if not using --key-file)')
def add_ssh_key(title: str, key_file: Optional[str], key: Optional[str]):
    """Add SSH key"""
    try:
        if key_file and key:
            click.echo("❌ Please specify either --key-file or --key, not both")
            exit(1)

        if key_file:
            if not os.path.exists(key_file):
                click.echo(f"❌ Key file not found: {key_file}")
                exit(1)
            with open(key_file, 'r') as f:
                key = f.read().strip()
        elif not key:
            click.echo("❌ Please specify either --key-file or --key")
            exit(1)

        result = model_client.add_ssh_key(title, key)
        click.echo(f"✅ SSH key '{title}' added successfully!")
        if result.get('fingerprint'):
            click.echo(f"   Fingerprint: {result['fingerprint']}")

    except Exception as e:
        click.echo(f"❌ Failed to add SSH key: {e}")
        exit(1)


@ssh.command("list")
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def list_ssh_keys(output_format: str):
    """List all SSH keys"""
    try:
        result = model_client.get_ssh_keys()
        keys = result.get('ssh_keys', [])

        if not keys:
            click.echo("No SSH keys found")
            return

        if output_format == 'json':
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🔑 Found {result.get('total', 0)} SSH keys:\n")
            for ssh_key in keys:
                click.echo(f"  • {ssh_key['title']} (ID: {ssh_key['id']})")
                if ssh_key.get('fingerprint'):
                    click.echo(f"    Fingerprint: {ssh_key['fingerprint']}")
                click.echo(f"    Created: {ssh_key['created_at']}")
                click.echo()

    except Exception as e:
        click.echo(f"❌ Failed to list SSH keys: {e}")
        exit(1)


@ssh.command("delete")
@click.argument('key_id')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def delete_ssh_key(key_id: str, confirm: bool):
    """Delete SSH key"""
    try:
        if not confirm:
            if not click.confirm(f"Are you sure you want to delete SSH key '{key_id}'?"):
                click.echo("Deletion cancelled")
                return

        model_client.delete_ssh_key(key_id)
        click.echo(f"✅ SSH key '{key_id}' deleted successfully!")

    except Exception as e:
        click.echo(f"❌ Failed to delete SSH key: {e}")
        exit(1)


# Inference Management Commands
@click.group()
def inference():
    """Inference management commands"""
    pass


@inference.command("gpus")
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def list_gpus(output_format: str):
    """List available GPUs"""
    try:
        result = model_client.get_available_gpus()
        gpus = result.get('gpus', [])

        if output_format == 'json':
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🖥️  Available GPUs ({result.get('total', 0)}):\n")
            for gpu in gpus:
                click.echo(f"  • {gpu.get('InstanceType', 'N/A')}")
                click.echo(f"    Model: {gpu.get('GpuModel', 'N/A')}")
                click.echo(f"    Count: {gpu.get('GpuCount', 'N/A')}")
                click.echo(f"    Memory: {gpu.get('MemoryGiB', 'N/A')} GiB")
                click.echo(f"    Price: €{gpu.get('PriceEUR', 'N/A')}")
                click.echo(f"    Region: {gpu.get('Region', 'N/A')}")
                click.echo()

    except Exception as e:
        click.echo(f"❌ Failed to list GPUs: {e}")
        exit(1)

def parse_vllm_config_file(file_path: str) -> dict:
    """Parse vLLM configuration from JSON file"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise click.ClickException(f"vLLM config file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in vLLM config file: {e}")

def build_vllm_config_from_options(**kwargs) -> dict:
    """Build vLLM config dict from command line options, excluding None values"""
    config = {}

    # Map CLI option names to vLLM config keys
    option_mapping = {
        'max_model_len': 'max_model_len',
        'max_num_batched_tokens': 'max_num_batched_tokens',
        'gpu_memory_utilization': 'gpu_memory_utilization',
        'tensor_parallel_size': 'tensor_parallel_size',
        'disable_log_requests': 'disable_log_requests',
        'enable_auto_tool_choice': 'enable_auto_tool_choice',
        'tool_call_parser': 'tool_call_parser',
        'chat_template': 'chat_template'
    }

    for cli_key, config_key in option_mapping.items():
        value = kwargs.get(cli_key)
        if value is not None:
            config[config_key] = value

    return config


@inference.command("load")
@click.argument('model_name')
@click.option('--gpu', help='GPU identifier (will show available GPUs if not specified)')
@click.option('--force', is_flag=True, help='Force loading even if model already loaded')
@click.option('--vllm-config-file', help='Path to JSON file containing vLLM configuration')
@click.option('--max-model-len', type=int, help='Maximum sequence length for the model')
@click.option('--max-num-batched-tokens', type=int, help='Maximum number of batched tokens')
@click.option('--gpu-memory-utilization', type=float, help='GPU memory utilization ratio (0.0-1.0)')
@click.option('--tensor-parallel-size', type=int, help='Number of GPUs to use for tensor parallelism')
@click.option('--disable-log-requests', is_flag=True, help='Disable logging of requests')
@click.option('--enable-auto-tool-choice', is_flag=True, help='Enable automatic tool choice')
@click.option('--tool-call-parser', help='Tool call parser to use (e.g., llama3_json)')
@click.option('--chat-template', help='Path to chat template file')
def load_model(model_name: str, gpu: Optional[str], force: bool, vllm_config_file: Optional[str], **vllm_options):
    """Load model for inference with optional vLLM configuration"""
    try:

        if vllm_config_file:
            # Load from file
            vllm_config = parse_vllm_config_file(vllm_config_file)
            click.echo(f"Loaded vLLM config from: {vllm_config_file}")
        else:
            # Build from command line options
            vllm_config = build_vllm_config_from_options(**vllm_options)
            if vllm_config:
                click.echo("Using vLLM config from command line options")

        # Show vLLM config if present
        if vllm_config:
            click.echo("vLLM Configuration:")
            for key, value in vllm_config.items():
                click.echo(f"  {key}: {value}")
            click.echo()

        # If no GPU specified, show available GPUs
        if not gpu:
            # For now, we'll use predefined GPU types since the API structure isn't clear
            gpus_result = model_client.get_available_gpus()
            gpus = gpus_result.get('gpus', [])
            available_gpu_types = [gpu.get('InstanceType') for gpu in gpus]

            click.echo("Available GPU types:")
            for i, gpu_type in enumerate(available_gpu_types):
                click.echo(f"  {i}: {gpu_type}")

            gpu_choice = click.prompt("Select GPU number", type=int)
            if gpu_choice < 0 or gpu_choice >= len(available_gpu_types):
                click.echo("Invalid GPU selection")
                exit(1)

            gpu = available_gpu_types[gpu_choice]

        result = model_client.load_model_for_inference(model_name, gpu, force, vllm_config)
        click.echo(f"Model '{model_name}' loading initiated on {gpu}!")

        if result.get('estimated_time'):
            click.echo(f"Estimated loading time: {result['estimated_time']} seconds")

    except Exception as e:
        click.echo(f"Failed to load model: {e}")
        exit(1)


@inference.command("unload")
@click.option('--session-id', type=int, help='Session ID to unload (will show loaded models if not specified)')
@click.option('--model-name', help='Unload by model name (alternative to session-id)')
def unload_model(session_id: Optional[int], model_name: Optional[str]):
    """Unload model from inference"""
    try:
        # If no session ID specified, show loaded models
        if not session_id and not model_name:
            loaded_result = model_client.get_loaded_models()
            loaded_models = loaded_result.get('loaded_models', [])

            if not loaded_models:
                click.echo("No models currently loaded")
                return

            click.echo("Currently loaded models:")
            for model in loaded_models:
                click.echo(f"  Session {model.get('session_id')}: {model.get('model_name')} on {model.get('gpu_id')}")

            session_choice = click.prompt("Enter session ID to unload", type=int)
            session_id = session_choice
        elif model_name and not session_id:
            # Find session ID by model name
            loaded_result = model_client.get_loaded_models()
            loaded_models = loaded_result.get('loaded_models', [])

            matching_sessions = [m for m in loaded_models if m.get('model_name') == model_name]
            if not matching_sessions:
                click.echo(f"❌ Model '{model_name}' is not currently loaded")
                exit(1)
            elif len(matching_sessions) > 1:
                click.echo(f"Multiple sessions found for '{model_name}':")
                for model in matching_sessions:
                    click.echo(f"  Session {model.get('session_id')}: on {model.get('gpu_id')}")
                session_id = click.prompt("Enter session ID to unload", type=int)
            else:
                session_id = matching_sessions[0].get('session_id')

        model_client.unload_model_from_inference(session_id)
        click.echo(f"✅ Model unloaded successfully! (Session {session_id})")

    except Exception as e:
        click.echo(f"❌ Failed to unload model: {e}")
        exit(1)


@inference.command("status")
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def inference_status(output_format: str):
    """Show currently loaded models"""
    try:
        result = model_client.get_loaded_models()
        loaded_models = result.get('loaded_models', [])

        if output_format == 'json':
            click.echo(json.dumps(result, indent=2))
        else:
            if not loaded_models:
                click.echo("No models currently loaded for inference")
                return

            click.echo(f"\n🚀 Currently loaded models ({result.get('total', 0)}):\n")
            for model in loaded_models:
                click.echo(f"  • {model.get('model_name')} (Session {model.get('session_id')})")
                click.echo(f"    GPU: {model.get('gpu_id')}")
                click.echo(f"    Loaded: {model.get('load_time')}")
                click.echo(f"    cost_from_start: {model.get('cost')}")

    except Exception as e:
        click.echo(f"❌ Failed to get inference status: {e}")
        exit(1)

# Workflow Management Commands
@click.group()
def workflow():
    """Inference management commands"""
    pass

# Helper Commands
@workflow.command("workflow")
@click.argument('model_name')
@click.option('--type', 'model_type', type=click.Choice(['huggingface', 'custom']),
              required=True, help='Type of model')
@click.option('--url', help='HuggingFace URL (required for huggingface models)')
@click.option('--api-key', help='HuggingFace API key (optional)')
@click.option('--ssh-key-file', help='Path to SSH public key file (for custom models)')
@click.option('--ssh-key-title', help='Title for SSH key (for custom models)')
@click.option('--local-model-path', help='Path to local model files (for custom models)')
@click.option('--auto-load', is_flag=True, help='Automatically load model for inference after upload')
def complete_workflow(model_name: str, model_type: str, url: Optional[str],
                      api_key: Optional[str], ssh_key_file: Optional[str], ssh_key_title: Optional[str],
                      local_model_path: Optional[str], auto_load: bool):
    """Complete workflow: register model, upload (if custom), and optionally load for inference"""

    try:
        click.echo(f"🚀 Starting complete workflow for '{model_name}'...")

        # Step 1: Register model
        click.echo("\n📝 Step 1: Registering model...")
        not_ssh = model_type == 'huggingface' or 'ollama'

        if not_ssh and not url:
            click.echo("❌ URL is required for HuggingFace models")
            exit(1)

        model_client.register_model(
            name=model_name,
            provider=model_type,
            url=url,
            api_key=api_key
        )
        click.echo("✅ Model registered successfully!")

        # Step 2: For custom models, handle SSH and upload
        if model_type == 'custom':
            click.echo("\n🔑 Step 2: Setting up SSH access...")

            # Add SSH key if provided
            if ssh_key_file and ssh_key_title:
                if os.path.exists(ssh_key_file):
                    with open(ssh_key_file, 'r') as f:
                        ssh_key_content = f.read().strip()

                    try:
                        model_client.add_ssh_key(ssh_key_title, ssh_key_content)
                        click.echo(f"✅ SSH key '{ssh_key_title}' added successfully!")
                    except Exception as e:
                        if "already exists" in str(e).lower():
                            click.echo(f"ℹ️  SSH key already exists, continuing...")
                        else:
                            raise e
                else:
                    click.echo(f"❌ SSH key file not found: {ssh_key_file}")
                    exit(1)

            # If local model path provided, show git instructions
            if local_model_path:
                click.echo(f"\n📂 Step 3: Upload model files...")
                click.echo("To upload your model files, run the following commands:")
                click.echo(f"")
                click.echo(f"  git clone git@gitlab.regolo.ai:<username>/{model_name}.git")
                click.echo(f"  cd {model_name}")
                click.echo(f"  cp -r {local_model_path}/* .")
                click.echo(f"  git add .")
                click.echo(f'  git commit -m "Add model files"')
                click.echo(f"  git push origin main")
                click.echo(f"")

                if click.confirm("Have you completed the git upload?"):
                    click.echo("✅ Model files uploaded!")
                else:
                    click.echo("⏸️  Workflow paused. Complete the git upload and then run the load command manually.")
                    return

        # Step 3: Auto-load for inference if requested
        if auto_load:
            click.echo(f"\n🖥️  Step 3: Loading model for inference...")

            # Get available GPUs
            gpus_result = model_client.get_available_gpus()
            gpus = gpus_result.get('gpus', [])

            if not gpus:
                click.echo("❌ No GPUs available for inference")
                return

            # Use the first available GPU or let user choose
            if len(gpus) == 1:
                gpu = gpus[0].get('InstanceType', 'GPU-0')
                click.echo(f"Using GPU: {gpu}")
            else:
                click.echo("Available GPUs:")
                for i, gpu_info in enumerate(gpus):
                    click.echo(f"  {i}: {gpu_info.get('InstanceType', 'N/A')} - {gpu_info.get('GpuModel', 'N/A')}")

                gpu_choice = click.prompt("Select GPU number", type=int)
                if gpu_choice < 0 or gpu_choice >= len(gpus):
                    click.echo("❌ Invalid GPU selection")
                    return

                gpu = gpus[gpu_choice].get('InstanceType', f'GPU-{gpu_choice}')

            # Load model for inference
            model_client.load_model_for_inference(model_name, gpu)
            click.echo(f"✅ Model '{model_name}' loading initiated on {gpu}!")

        click.echo(f"\n🎉 Workflow completed successfully!")
        click.echo(f"   Model: {model_name}")
        click.echo(f"   Type: {model_type}")
        click.echo(f"   Project: {model_name}")
        if auto_load:
            click.echo(f"   Status: Loading for inference...")

    except Exception as e:
        click.echo(f"❌ Workflow failed: {e}")
        exit(1)


@inference.command("user-status")
@click.option('--month', help='Filter by month in MMYYYY format (e.g., 012025 for January 2025)')
@click.option('--time-range-start', help='Start of time range, ISO 8601 format')
@click.option('--time-range-end', help='End of time range, ISO 8601 format')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']),
              default='table', help='Output format')
def user_inference_status(month: Optional[str], time_range_start: Optional[str], time_range_end: Optional[str], output_format: str):
    """Get inference status for your models"""
    try:
        # Validate month format if provided
        if month and not re.match(r'^(0[1-9]|1[0-2])\d{4}$', month):
            click.echo("❌ Invalid month format. Use MMYYYY format (e.g., 012025)")
            exit(1)

        result = model_client.get_user_inference_status(month, time_range_start, time_range_end)

        if output_format == 'json':
            click.echo(json.dumps(result, indent=2))
        else:
            if month:
                click.echo(f"\n📊 Your Inference Status for {month}:\n")
            else:
                click.echo(f"\n📊 Your Currently Running Models:\n")

            # Display the status information
            click.echo(pprint.pformat(result))
            """else:
                click.echo("  No models found")"""

    except Exception as e:
        click.echo(f"❌ Failed to get user inference status: {e}")
        exit(1)

# Existing commands (keeping them)
@click.command("get-available-models", help="Gets available models")
@click.option('--api-key', required=True, help='The API key used to query Regolo.')
@click.option('--model-type', default="", required=False,
              type=click.Choice(['', 'chat', 'image_generation', "embedding", "audio_transcription", "rerank"]),
              help='The type of the models you want to retrieve (returns all by default)')
def get_available_models(api_key: str, model_type: str):
    available_models: list[dict] = regolo.RegoloClient.get_available_models(api_key, model_info=True)
    output_models: list[tuple] = []
    for model in available_models:
        if model_type in model["model_info"]["mode"]:
            output_models.append((model["model_name"], model["model_info"]["mode"]))
    click.echo(pprint.pformat(output_models))


@click.command("chat", help="Allows chatting with LLMs")
@click.option('--no-hide', required=False, is_flag=True, default=False, help='Do not hide the API key when typing')
@click.option('--api-key', required=False, help='The API key used to chat with Regolo.')
@click.option('--disable-newlines', required=False, is_flag=True, default=False,
              help='Disable new lines, they will be replaced with space character')
def chat(no_hide: bool, api_key: str, disable_newlines: bool):
    if not api_key:
        api_key = click.prompt("Insert your regolo API key", hide_input=not no_hide)
    available_models: list[dict] = regolo.RegoloClient.get_available_models(api_key, model_info=True)

    if len(available_models) == 0:
        click.echo("No models available with your API key.")
        exit(1)

    available_models_dict = {}
    model_number = 1
    for model in available_models:
        mode = model["model_info"]["mode"]
        match mode:
            case None:
                pass
            case "chat":
                available_models_dict[model_number] = model
                available_models_dict[model_number] = [model["model_name"], mode]
                model_number += 1

    click.echo(f"The models you can use are:\n {pprint.pformat(available_models_dict)}")

    prompt_model_number = click.prompt("Write the number of the model to use")

    try:
        model = available_models_dict[int(prompt_model_number)][0]
    except ValueError:
        raise Exception("Not a number")
    except KeyError:
        raise Exception("Not a valid number.")

    click.echo(f"\n")

    client = RegoloClient(api_key=api_key, chat_model=model)
    click.echo(f"You can now chat with {model}, write \"/bye\" to exit")

    while True:
        # ask user input
        user_input = click.prompt("user")

        if user_input == "/bye":
            exit(0)
        # get chat response and save in the client
        response = client.run_chat(user_input, stream=True, full_output=False)

        # print output
        while True:
            try:
                res = next(response)
                if res[0]:
                    click.echo(res[0] + ":")
                if disable_newlines:
                    res[1] = res[1].replace("\n", " ")
                click.echo(res[1], nl=False)
            except StopIteration:
                break

        click.echo("\n")


@click.command("create-image", help='Creates images')
@click.option('--api-key', required=True, help='The API key used generate.')
@click.option('--model', required=True, help="The number of images to generate. (Defaults to 1)")
@click.option('--save-path', help='The path in which to save the images. (Defaults to ../images)')
@click.option('--prompt', default="A generic image",
              help='The text prompt for image generation. (Defaults to "A generic image")')
@click.option('--n', default=1, help='The number of images to generate. (Defaults to 1)')
@click.option('--quality', default="standard",
              help="The quality of the image that will be generated. The 'hd' value creates images with finer details and greater consistency across the image. (Defaults to 'standard'')")
@click.option('--size', default="1024x1024", help="The size of the generated images. (Defaults to '1024x1024')")
@click.option('--style', default="realistic", help="The style of the generated images. (Defaults to 'realistic')")
@click.option('--output-file-format', default="png", type=click.Choice(['png', 'jpg', 'jpeg', 'webp', 'bmp'], case_sensitive=False),
              help="The output file format for the generated images. (Defaults to 'png')")
def create_image(api_key: str, save_path: str, model: str, prompt: str, n: int, quality: str, size: str, style: str, output_file_format: str):
    if model is None:
        raise Exception("You must specify a model")

    KeysHandler.check_key(api_key)

    if save_path is None:
        save_path = os.path.join(os.getcwd(), "images")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    client = RegoloClient(api_key=api_key, image_generation_model=model)

    images_bytes = client.create_image(prompt=prompt, n=n, quality=quality, size=size, style=style)

    # Normalize format extension
    ext = output_file_format.lower()
    if ext == 'jpg':
        ext = 'jpeg'

    for image_bytes in images_bytes:
        image = Image.open(BytesIO(image_bytes))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(save_path, f"{timestamp}.{ext}")

        i = 1
        while os.path.exists(filepath):
            filename = f"{timestamp}_{i}.{ext}"
            filepath = os.path.join(save_path, filename)
            i += 1

        # Save with appropriate format
        if ext == 'jpeg':
            image.save(filepath, 'JPEG')
        else:
            image.save(filepath, ext.upper())


@click.command("transcribe-audio", help='Transcribes audio files')
@click.option('--api-key', required=True, help='The API key used to transcribe.')
@click.option('--model', required=True,
              help='The model to use for transcription (gpt-4o-transcribe, gpt-4o-mini-transcribe, or whisper-1)')
@click.option('--file-path', required=True, help='Path to the audio file to transcribe')
@click.option('--save-path', help='Path to save the transcription (prints to console if not specified)')
@click.option('--language', help='Language of the input audio in ISO-639-1 format (e.g., en, es, fr)')
@click.option('--prompt', help='Optional text to guide the model\'s style or continue a previous audio segment')
@click.option('--response-format', default="json", type=click.Choice(['json', 'text', 'srt', 'verbose_json', 'vtt']),
              help='Format of the transcript output (defaults to json)')
@click.option('--temperature', type=float, help='Sampling temperature between 0 and 1 (defaults to 0)')
@click.option('--chunking-strategy', help='Controls audio chunking: "auto" or JSON object string')
@click.option('--include-logprobs', is_flag=True,
              help='Include log probabilities (only works with json format and gpt-4o models)')
@click.option('--timestamp-granularities', multiple=True, type=click.Choice(['word', 'segment']),
              help='Timestamp granularities (requires verbose_json format)')
@click.option('--stream', is_flag=True, help='Stream the response (not supported for whisper-1)')
@click.option('--full-output', is_flag=True, help='Return full API response instead of just text')
def transcribe_audio(api_key: str, model: str, file_path: str, save_path: str, language: str,
                     prompt: str, response_format: str, temperature: float, chunking_strategy: str,
                     include_logprobs: bool, timestamp_granularities: tuple, stream: bool, full_output: bool):
    # Validate API key
    KeysHandler.check_key(api_key)

    # Prepare optional parameters
    kwargs: dict[Any, Any] = {
        'file': file_path,
        'model': model,
        'api_key': api_key,
        'response_format': response_format,
        'stream': stream,
        'full_output': full_output
    }

    if language:
        kwargs['language'] = language
    if prompt:
        kwargs['prompt'] = prompt
    if temperature is not None:
        kwargs['temperature'] = temperature
    if chunking_strategy:
        kwargs['chunking_strategy'] = chunking_strategy
    if include_logprobs:
        kwargs['include'] = ['logprobs']
    if timestamp_granularities:
        kwargs['timestamp_granularities'] = list(timestamp_granularities)

    # Create client and transcribe
    client = RegoloClient(api_key=api_key)

    try:
        if stream:
            # Handle streaming
            click.echo("Transcribing (streaming)...")
            response = client.static_audio_transcription(**kwargs)

            if save_path:
                # Stream to file
                with open(save_path, 'w', encoding='utf-8') as f:
                    for chunk in response:
                        if chunk:
                            f.write(str(chunk))
                            f.flush()
                click.echo(f"Transcription saved to: {save_path}")
            else:
                # Stream to console
                for chunk in response:
                    if chunk:
                        click.echo(str(chunk), nl=False)
                click.echo()  # Final newline
        else:
            # Handle non-streaming
            click.echo("Transcribing...")
            response = client.static_audio_transcription(**kwargs)

            # Format output based on response_format and full_output
            if full_output:
                output = json.dumps(response, indent=2, ensure_ascii=False)
            elif response_format == 'json' and isinstance(response, dict):
                output = response.get('text', str(response))
            else:
                output = str(response)

            if save_path:
                # Save to file
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(output)
                click.echo(f"Transcription saved to: {save_path}")
            else:
                # Print to console
                click.echo(output)

    except Exception as e:
        raise click.ClickException(f"Transcription failed: {str(e)}")


@click.command("rerank", help='Rerank documents based on relevance to a query')
@click.option('--api-key', required=True, help='The API key used for reranking.')
@click.option('--model', required=True, help='The reranking model to use (e.g., jina-reranker-v2, cohere-rerank-v3)')
@click.option('--query', required=True, help='The search query to compare documents against')
@click.option('--documents', required=True, multiple=True,
              help='Documents to rerank (can be specified multiple times)')
@click.option('--documents-file', help='Path to JSON file containing documents array')
@click.option('--top-n', type=int, help='Number of most relevant documents to return (returns all if not specified)')
@click.option('--rank-fields', multiple=True,
              help='For structured documents, specify which fields to rank by (can be specified multiple times)')
@click.option('--no-return-documents', is_flag=True, default=False,
              help='Do not return document content in results (only indices and scores)')
@click.option('--max-chunks-per-doc', type=int, help='Maximum number of chunks per document')
@click.option('--save-path', help='Path to save the reranking results as JSON (prints to console if not specified)')
@click.option('--full-output', is_flag=True, help='Return full API response instead of just results')
@click.option('--format', 'output_format', type=click.Choice(['json', 'table']),
              default='table', help='Output format (defaults to table)')
def rerank_documents(api_key: str, model: str, query: str, documents: tuple, documents_file: str,
                     top_n: int, rank_fields: tuple, no_return_documents: bool,
                     max_chunks_per_doc: int, save_path: str, full_output: bool, output_format: str):
    """Rerank documents based on their relevance to a query"""

    # Validate API key
    KeysHandler.check_key(api_key)

    # Prepare documents list
    docs_list = []

    if documents_file:
        # Load documents from JSON file
        try:
            with open(documents_file, 'r', encoding='utf-8') as f:
                docs_list = json.load(f)

            if not isinstance(docs_list, list):
                raise click.ClickException("Documents file must contain a JSON array")

        except FileNotFoundError:
            raise click.ClickException(f"Documents file not found: {documents_file}")
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in documents file: {e}")
    else:
        # Use documents from command line
        if not documents:
            raise click.ClickException("Either --documents or --documents-file must be specified")
        docs_list = list(documents)

    if not docs_list:
        raise click.ClickException("No documents provided for reranking")

    # Prepare optional parameters
    kwargs: dict[str, Any] = {
        'query': query,
        'documents': docs_list,
        'return_documents': not no_return_documents,
        'full_output': full_output
    }

    if top_n is not None:
        kwargs['top_n'] = top_n
    if rank_fields:
        kwargs['rank_fields'] = list(rank_fields)
    if max_chunks_per_doc is not None:
        kwargs['max_chunks_per_doc'] = max_chunks_per_doc

    # Create client and rerank
    client = RegoloClient(api_key=api_key, reranker_model=model)

    try:
        click.echo(f"Reranking {len(docs_list)} documents with query: '{query}'...")

        try:
            response = client.rerank(
                **kwargs
            )
        except Exception as e:
            print(e)
            raise click.ClickException(f"Reranking failed: {str(e)}")
        # Format output
        if output_format == 'json':
            output = json.dumps(response, indent=2, ensure_ascii=False)
        else:
            # Table format
            if full_output and isinstance(response, dict):
                results = response.get('results', [])
                metadata = {k: v for k, v in response.items() if k != 'results'}

                output_lines = []
                if metadata:
                    output_lines.append("Response Metadata:")
                    output_lines.append(json.dumps(metadata, indent=2))
                    output_lines.append("\n")
            else:
                results = response
                output_lines = []

            output_lines.append(f"\n📊 Reranking Results (Top {len(results)} documents):\n")

            for i, result in enumerate(results, 1):
                index = result.get('index', 'N/A')
                score = result.get('relevance_score', 0.0)

                # Format score as percentage
                score_pct = f"{score * 100:.2f}%" if isinstance(score, (int, float)) else str(score)

                output_lines.append(f"  {i}. Document #{index} - Relevance: {score_pct}")

                if 'document' in result:
                    doc = result['document']
                    if isinstance(doc, dict):
                        # Structured document
                        output_lines.append(f"     Content: {json.dumps(doc, ensure_ascii=False)}")
                    else:
                        # String document - truncate if too long
                        doc_str = str(doc)
                        if len(doc_str) > 200:
                            doc_str = doc_str[:197] + "..."
                        output_lines.append(f"     Content: {doc_str}")

                output_lines.append("")

            output = "\n".join(output_lines)

        # Save or print output
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                if output_format == 'json':
                    f.write(output)
                else:
                    # Save as JSON even if displayed as table
                    f.write(json.dumps(response, indent=2, ensure_ascii=False))
            click.echo(f"✅ Reranking results saved to: {save_path}")
        else:
            click.echo(output)

    except Exception as e:
        raise click.ClickException(f"Reranking failed: {str(e)}")


# Add all command groups to CLI
cli.add_command(auth)
cli.add_command(models)
cli.add_command(ssh)
cli.add_command(inference)
cli.add_command(workflow)

# Add inference commands
cli.add_command(transcribe_audio)
cli.add_command(chat)
cli.add_command(create_image)
cli.add_command(get_available_models)
cli.add_command(rerank_documents)

if __name__ == '__main__':
    cli()
