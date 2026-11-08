import json
import os
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
from regolo.management import VLLM_FIELDS
from regolo.management import ModelManagementClient as _ManagementClient

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
AUDIO_EXTENSIONS = ["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"]

# Configuration file for storing auth tokens
CONFIG_FILE = os.path.expanduser(os.getenv("REGOLO_CONFIG_FILE", "~/.regolo_config.json"))


class ModelManagementClient(_ManagementClient):
    """Backward-compatible CLI import; SDK users should import from regolo."""

    def __init__(self, base_url="https://devmid.regolo.ai", **kwargs):
        kwargs.setdefault("config_file", CONFIG_FILE)
        super().__init__(base_url, **kwargs)


# Management commands initialize this lazily; inference imports read no token file.
model_client: ModelManagementClient | None = None


def _get_model_client() -> ModelManagementClient:
    if model_client is None:
        raise click.ClickException("Management client is not initialized")
    return model_client


@click.option("--devmid-url", envvar="DEVMID_URL", help="Devmid server URL for management commands")
@click.option(
    "--config-file", type=click.Path(), envvar="REGOLO_CONFIG_FILE", help="Server-scoped token file"
)
@click.option(
    "--public-openapi",
    type=click.Path(exists=True, dir_okay=False),
    envvar="DEVMID_PUBLIC_OPENAPI",
    help="Current exported public Devmid OpenAPI JSON",
)
@click.pass_context
def _cli(
    ctx: click.Context,
    devmid_url: str | None = None,
    config_file: str | None = None,
    public_openapi: str | None = None,
):
    global model_client
    if ctx.invoked_subcommand not in {"auth", "models", "ssh", "inference", "workflow"}:
        return
    if (
        model_client is None
        or model_client.is_closed
        or devmid_url is not None
        or config_file is not None
        or public_openapi is not None
    ):
        previous = model_client
        base_url = devmid_url or (
            previous.base_url
            if previous is not None
            else os.getenv("DEVMID_URL", "https://devmid.regolo.ai")
        )
        path = config_file or (previous.config_file if previous is not None else CONFIG_FILE)
        try:
            initialized_client = ModelManagementClient(
                base_url, config_file=path, public_openapi=public_openapi
            )
        except (ValueError, OSError) as exc:
            raise click.ClickException(str(exc)) from exc
        model_client = initialized_client
        ctx.call_on_close(initialized_client.close)
        if previous is not None:
            previous.close()


cli: click.Group = click.group()(_cli)


# Authentication Commands
@click.group()
def auth():
    """Authentication commands"""
    pass


@auth.command("login")
@click.option("--username", prompt=True, help="Username for authentication")
@click.option("--password", prompt=True, hide_input=True, help="Password for authentication")
def login(username: str, password: str):
    """Login and save authentication tokens"""
    try:
        result = _get_model_client().authenticate(username, password)
        click.echo(
            f"✅ Successfully authenticated! Token expires in {result.get('expires_in', 'unknown')} seconds."
        )
    except Exception as e:
        click.echo(f"❌ Authentication failed: {e}")
        exit(1)


@auth.command("logout")
def logout():
    """Logout and clear saved tokens"""
    _get_model_client().logout()
    click.echo("✅ Local tokens for this server cleared.")


# Model Management Commands
@click.group()
def models():
    """Model management commands"""
    pass


@models.command("register")
@click.option("--name", required=True, help="Name for the model")
@click.option(
    "--type",
    "model_type",
    type=click.Choice(["huggingface"]),
    default="huggingface",
    show_default=True,
    help="Model provider",
)
@click.option("--url", help="HuggingFace URL (required for huggingface models)")
@click.option("--api-key", help="HuggingFace API key (optional, for private models)")
@click.option("--revision", help="Commit, tag, or branch (defaults to latest)")
def register_model(
    name: str, model_type: str, url: Optional[str], api_key: Optional[str], revision: Optional[str]
):
    """Register a new model in the system"""
    try:
        if model_type == "huggingface" and not url:
            click.echo("❌ URL is required for HuggingFace models")
            exit(1)

        registered = _get_model_client().register_model(
            name=name,
            provider=model_type,
            url=url,
            api_key=api_key,
            revision=revision,
        )

        click.echo(registered.get("message") or f"✅ Model '{name}' registered successfully!")
        click.echo("🔍 The immutable revision is queued for security scanning.")

    except Exception as e:
        click.echo(f"❌ Failed to register model: {e}")
        exit(1)


@models.command("list")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list_models(output_format: str):
    """List all registered models"""
    try:
        result = _get_model_client().get_models()
        models_list = result.get("models") or []

        if output_format != "json" and not models_list:
            click.echo("No models found")
            return

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n📋 Found {result.get('total', 0)} models:\n")
            for model in models_list:
                model_type = model["provider"]
                click.echo(f"  • {model['name']} ({model_type})")
                if model.get("url"):
                    click.echo(f"    URL: {model['url']}")
                click.echo(f"    Registration ID: {model.get('registration_id')}")
                click.echo(f"    Artifact ID: {model.get('scanned_artifact_id')}")
                click.echo(f"    Revision: {model.get('revision') or model.get('source_revision')}")
                click.echo(f"    Scan: {model.get('scan_status')}")
                if model.get("content_digest"):
                    click.echo(f"    Content digest: {model['content_digest']}")
                if model.get("scan_detail"):
                    click.echo(f"    Scan detail: {model['scan_detail']}")
                click.echo(f"    Created: {model['created_at']}")
                click.echo()

    except Exception as e:
        click.echo(f"❌ Failed to list models: {e}")
        exit(1)


@models.command("details")
@click.argument("model_name")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def model_details(model_name: str, output_format: str):
    """Get detailed information about a specific model"""
    try:
        model = _get_model_client().get_model(model_name)

        if output_format == "json":
            click.echo(json.dumps(model, indent=2))
        else:
            model_type = model["provider"]
            click.echo(f"\n📋 Model Details: {model['name']}")
            click.echo(f"  Type: {model_type}")
            click.echo(f"  Email: {model['email']}")
            if model.get("url"):
                click.echo(f"  URL: {model['url']}")
            click.echo(f"  Registration ID: {model.get('registration_id') or '-'}")
            click.echo(f"  Artifact ID: {model.get('scanned_artifact_id') or '-'}")
            click.echo(
                f"  Revision: {model.get('revision') or model.get('source_revision') or '-'}"
            )
            click.echo(f"  Content digest: {model.get('content_digest') or '-'}")
            click.echo(f"  Scan status: {model.get('scan_status') or '-'}")
            click.echo(f"  Scan verdict: {model.get('scan_verdict') or '-'}")
            click.echo(f"  Scan detail: {model.get('scan_detail') or '-'}")
            click.echo(f"  Created: {model['created_at']}")

    except Exception as e:
        click.echo(f"❌ Failed to get model details: {e}")
        exit(1)


@models.command("delete")
@click.argument("model_name")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def delete_model(model_name: str, confirm: bool):
    """Delete a model"""
    try:
        if not confirm:
            if not click.confirm(f"Are you sure you want to delete model '{model_name}'?"):
                click.echo("Deletion cancelled")
                return

        _get_model_client().delete_model(model_name)
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
@click.option("--title", required=True, help="Title for the SSH key")
@click.option("--key-file", help="Path to SSH public key file")
@click.option("--key", help="SSH public key content (if not using --key-file)")
def add_ssh_key(title: str, key_file: Optional[str], key: Optional[str]):
    """Add an SSH key"""
    try:
        if key_file and key:
            click.echo("❌ Please specify either --key-file or --key, not both")
            exit(1)

        if key_file:
            if not os.path.exists(key_file):
                click.echo(f"❌ Key file not found: {key_file}")
                exit(1)
            with open(key_file, "r") as f:
                key = f.read().strip()
        elif not key:
            click.echo("❌ Please specify either --key-file or --key")
            exit(1)

        result = _get_model_client().add_ssh_key(title, key)
        click.echo(f"✅ SSH key '{title}' added successfully!")
        if result.get("fingerprint"):
            click.echo(f"   Fingerprint: {result['fingerprint']}")

    except Exception as e:
        click.echo(f"❌ Failed to add SSH key: {e}")
        exit(1)


@ssh.command("list")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list_ssh_keys(output_format: str):
    """List all SSH keys"""
    try:
        result = _get_model_client().get_ssh_keys()
        keys = result.get("ssh_keys") or []

        if output_format != "json" and not keys:
            click.echo("No SSH keys found")
            return

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"\n🔑 Found {result.get('total', 0)} SSH keys:\n")
            for ssh_key in keys:
                click.echo(f"  • {ssh_key['title']} (ID: {ssh_key['id']})")
                if ssh_key.get("fingerprint"):
                    click.echo(f"    Fingerprint: {ssh_key['fingerprint']}")
                click.echo(f"    Created: {ssh_key['created_at']}")
                click.echo()

    except Exception as e:
        click.echo(f"❌ Failed to list SSH keys: {e}")
        exit(1)


@ssh.command("delete")
@click.argument("key_id")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def delete_ssh_key(key_id: str, confirm: bool):
    """Delete SSH key"""
    try:
        if not confirm:
            if not click.confirm(f"Are you sure you want to delete SSH key '{key_id}'?"):
                click.echo("Deletion cancelled")
                return

        _get_model_client().delete_ssh_key(key_id)
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
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list_gpus(output_format: str):
    """List available GPUs"""
    try:
        result = _get_model_client().get_available_gpus()
        gpus = result.get("gpus") or []

        if output_format == "json":
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
        with open(file_path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise click.ClickException("vLLM config must be a JSON object")
        return data
    except FileNotFoundError:
        raise click.ClickException(f"vLLM config file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in vLLM config file: {e}")


def build_vllm_config_from_options(**kwargs) -> dict:
    """Build vLLM config dict from command line options, excluding None values"""
    config = {}

    for key, value in kwargs.items():
        if value is None:
            continue
        if key not in VLLM_FIELDS:
            raise click.ClickException(f"{key} is unsupported or server-controlled in Devmid 3.9")
        config[key] = value

    return config


@inference.command("load")
@click.argument("model_name")
@click.option("--gpu", help="GPU identifier (will show available GPUs if not specified)")
@click.option("--force", is_flag=True, help="Force loading even if model already loaded")
@click.option("--vllm-config-file", help="Path to JSON file containing vLLM configuration")
@click.option("--max-model-len", type=int, help="Maximum sequence length for the model")
@click.option("--max-num-batched-tokens", type=int, help="Maximum number of batched tokens")
@click.option("--gpu-memory-utilization", type=float, help="GPU memory utilization ratio (0.0-1.0)")
@click.option(
    "--tensor-parallel-size", type=int, help="Number of GPUs to use for tensor parallelism"
)
@click.option(
    "--disable-log-requests", is_flag=True, default=None, help="Disable logging of requests"
)
@click.option(
    "--enable-auto-tool-choice", is_flag=True, default=None, help="Enable automatic tool choice"
)
@click.option("--tool-call-parser", help="Tool call parser to use (e.g., llama3_json)")
@click.option("--chat-template", help="Path to chat template file")
@click.option("--min-replicas", type=click.IntRange(min=0), help="Minimum replicas, including zero")
@click.option("--max-replicas", type=click.IntRange(min=1), help="Maximum replicas")
@click.option("--target-requests", type=click.IntRange(min=1))
@click.option("--scale-down-delay-seconds", type=click.IntRange(min=0))
@click.option("--engine", help="Engine declared by this server's public load schema")
@click.option(
    "--engine-config",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON engine tuning options",
)
@click.option(
    "--deployment-options",
    type=click.Path(exists=True, dir_okay=False),
    help="Additional public-schema deployment fields (e.g. GGUF selection)",
)
@click.option("--wait", "wait_ready", is_flag=True, help="Wait for runtime readiness")
@click.option("--timeout", type=click.FloatRange(min=0), default=1800, show_default=True)
def load_model(
    model_name: str,
    gpu: Optional[str],
    force: bool,
    vllm_config_file: Optional[str],
    min_replicas=None,
    max_replicas=None,
    target_requests=None,
    scale_down_delay_seconds=None,
    engine=None,
    engine_config=None,
    deployment_options=None,
    wait_ready=False,
    timeout=1800,
    **vllm_options,
):
    """Load model for inference with optional vLLM configuration"""
    try:
        if vllm_config_file and any(v is not None for v in vllm_options.values()):
            raise click.ClickException("Use either a vLLM config file or individual vLLM options")
        if vllm_config_file:
            # Load from a file
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
            # Offer only catalog entries with current capacity.
            gpus_result = _get_model_client().get_available_gpus()
            gpus: list[dict[str, Any]] = gpus_result.get("gpus") or []
            available_gpu_types: list[str] = [
                g["InstanceType"] for g in gpus if g.get("Available") is True
            ]

            if not available_gpu_types:
                raise click.ClickException("No GPU plans currently have capacity")
            click.echo("Available GPU types:")
            for i, gpu_type in enumerate(available_gpu_types):
                click.echo(f"  {i}: {gpu_type}")

            gpu_choice = click.prompt("Select GPU number", type=int)
            if gpu_choice < 0 or gpu_choice >= len(available_gpu_types):
                click.echo("Invalid GPU selection")
                exit(1)

            gpu = available_gpu_types[gpu_choice]

        plan = None
        if any(
            value is not None
            for value in (min_replicas, max_replicas, target_requests, scale_down_delay_seconds)
        ):
            plan = {
                "min_replicas": 1 if min_replicas is None else min_replicas,
                "max_replicas": 1 if max_replicas is None else max_replicas,
            }
            if target_requests is not None:
                plan["target_requests"] = target_requests
            if scale_down_delay_seconds is not None:
                plan["scale_down_delay_seconds"] = scale_down_delay_seconds
        assert gpu is not None
        result = _get_model_client().load_model_for_inference(
            model_name,
            gpu,
            force,
            vllm_config or None,
            plan,
            engine=engine,
            engine_config=parse_vllm_config_file(engine_config) if engine_config else None,
            deployment_options=parse_vllm_config_file(deployment_options)
            if deployment_options
            else None,
        )
        click.echo(json.dumps(result, indent=2))
        if wait_ready:
            click.echo(
                json.dumps(
                    _get_model_client().wait_for_inference(result["session_id"], timeout=timeout),
                    indent=2,
                )
            )
        click.echo(f"Model '{model_name}' loading initiated on {gpu}!")

        if result.get("estimated_time"):
            click.echo(f"Estimated loading time: {result['estimated_time']} seconds")

    except Exception as e:
        click.echo(f"Failed to load model: {e}")
        exit(1)


@inference.command("unload")
@click.argument("session_argument", required=False, type=click.IntRange(min=1))
@click.option(
    "--session-id", type=int, help="Session ID to unload (will show loaded models if not specified)"
)
@click.option("--model-name", help="Unload by model name (alternative to session-id)")
def unload_model(session_id: Optional[int], model_name: Optional[str], session_argument=None):
    """Unload model from inference"""
    try:
        if session_argument is not None:
            if session_id is not None or model_name is not None:
                raise click.ClickException("Use one session ID or model name")
            session_id = session_argument
        if session_id is not None and model_name is not None:
            raise click.ClickException("Use either --session-id or --model-name")
        # If no session ID specified, show loaded models
        if not session_id and not model_name:
            loaded_result = _get_model_client().get_loaded_models()
            loaded_models = loaded_result.get("loaded_models") or []

            if not loaded_models:
                click.echo("No models currently loaded")
                return

            click.echo("Currently loaded models:")
            for model in loaded_models:
                click.echo(
                    f"  Session {model.get('session_id')}: {model.get('model_name')} on {model.get('gpu_id')}"
                )

            session_choice = click.prompt("Enter session ID to unload", type=int)
            session_id = session_choice
        elif model_name and not session_id:
            # Find session ID by model name
            loaded_result = _get_model_client().get_loaded_models()
            loaded_models = loaded_result.get("loaded_models") or []

            matching_sessions = [m for m in loaded_models if m.get("model_name") == model_name]
            if not matching_sessions:
                click.echo(f"❌ Model '{model_name}' is not currently loaded")
                exit(1)
            elif len(matching_sessions) > 1:
                click.echo(f"Multiple sessions found for '{model_name}':")
                for model in matching_sessions:
                    click.echo(f"  Session {model.get('session_id')}: on {model.get('gpu_id')}")
                session_id = click.prompt("Enter session ID to unload", type=int)
            else:
                session_id = matching_sessions[0].get("session_id")

        if session_id is None:
            raise click.ClickException("Selected model has no session ID")
        _get_model_client().unload_model_from_inference(session_id)
        click.echo(f"✅ Model unloaded successfully! (Session {session_id})")

    except Exception as e:
        click.echo(f"❌ Failed to unload model: {e}")
        exit(1)


@inference.command("status")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
@click.option("--include-scaling", is_flag=True, help="Include bounded server scaling history")
def inference_status(output_format: str, include_scaling=False):
    """Show currently loaded models"""
    try:
        result = _get_model_client().get_loaded_models(include_scaling=include_scaling)
        loaded_models = result.get("loaded_models") or []

        if output_format == "json":
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
@click.argument("model_name")
@click.option(
    "--type",
    "model_type",
    type=click.Choice(["huggingface"]),
    default="huggingface",
    show_default=True,
    help="Model provider",
)
@click.option("--url", help="HuggingFace URL (required for huggingface models)")
@click.option("--api-key", help="HuggingFace API key (optional)")
@click.option("--revision", help="Commit, tag, or branch (defaults to latest)")
@click.option(
    "--auto-load", is_flag=True, help="Automatically load model for inference after upload"
)
@click.option(
    "--scan-timeout",
    type=float,
    default=3600,
    show_default=True,
    help="Seconds to wait for a safe scan before auto-loading",
)
def complete_workflow(
    model_name: str,
    model_type: str,
    url: Optional[str],
    api_key: Optional[str],
    revision: Optional[str],
    auto_load: bool,
    scan_timeout: float,
):
    """Register a Hugging Face revision and optionally load it after scanning."""

    try:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", model_name):
            raise click.ClickException(
                "Workflow names must use lowercase letters, digits and single hyphens; use register/list for server-normalized names"
            )
        click.echo(f"🚀 Starting complete workflow for '{model_name}'...")

        # Step 1: Register model
        click.echo("\n📝 Step 1: Registering model...")
        if not url:
            click.echo("❌ URL is required for HuggingFace models")
            exit(1)

        _get_model_client().register_model(
            name=model_name,
            provider=model_type,
            url=url,
            api_key=api_key,
            revision=revision,
        )
        click.echo("✅ Model registered successfully!")

        # Step 3: Auto-load for inference if requested
        if auto_load:
            click.echo("\n🔍 Waiting for the shared artifact scan to become safe...")
            _get_model_client().wait_for_scan(model_name, timeout=scan_timeout)
            click.echo("🖥️  Loading model for inference...")

            # Get available GPUs
            gpus_result = _get_model_client().get_available_gpus()
            gpus: list[dict[str, Any]] = [
                g for g in (gpus_result.get("gpus") or []) if g.get("Available") is True
            ]

            if not gpus:
                raise click.ClickException("No GPUs available for inference")

            # Use the first available GPU or let the user choose
            if len(gpus) == 1:
                gpu: str = gpus[0].get("InstanceType", "GPU-0")
                click.echo(f"Using GPU: {gpu}")
            else:
                click.echo("Available GPUs:")
                for i, gpu_info in enumerate(gpus):
                    click.echo(
                        f"  {i}: {gpu_info.get('InstanceType', 'N/A')} - {gpu_info.get('GpuModel', 'N/A')}"
                    )

                gpu_choice = click.prompt("Select GPU number", type=int)
                if gpu_choice < 0 or gpu_choice >= len(gpus):
                    raise click.ClickException("Invalid GPU selection")

                gpu = gpus[gpu_choice].get("InstanceType", f"GPU-{gpu_choice}")

            # Load model for inference
            _get_model_client().load_model_for_inference(model_name, gpu)
            click.echo(f"✅ Model '{model_name}' loading initiated on {gpu}!")

        click.echo("\n🎉 Workflow completed successfully!")
        click.echo(f"   Model: {model_name}")
        click.echo(f"   Type: {model_type}")
        click.echo(f"   Project: {model_name}")
        if auto_load:
            click.echo("   Status: Loading for inference...")

    except Exception as e:
        click.echo(f"❌ Workflow failed: {e}")
        exit(1)


@inference.command("user-status")
@click.option("--month", help="Filter by month in MMYYYY format (e.g., 012025 for January 2025)")
@click.option("--time-range-start", help="Start of time range, ISO 8601 format")
@click.option("--time-range-end", help="End of time range, ISO 8601 format")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def user_inference_status(
    month: Optional[str],
    time_range_start: Optional[str],
    time_range_end: Optional[str],
    output_format: str,
):
    """Get inference status for your models"""
    try:
        # Validate a month format if provided
        if month and not re.match(r"^(0[1-9]|1[0-2])\d{4}$", month):
            click.echo("❌ Invalid month format. Use MMYYYY format (e.g., 012025)")
            exit(1)

        result = _get_model_client().get_user_inference_status(
            month, time_range_start, time_range_end
        )

        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            if month:
                click.echo(f"\n📊 Your Inference Status for {month}:\n")
            else:
                click.echo("\n📊 Your Currently Running Models:\n")

            # Display the status information
            click.echo(pprint.pformat(result))
            """else:
                click.echo("  No models found")"""

    except Exception as e:
        click.echo(f"❌ Failed to get user inference status: {e}")
        exit(1)


# Existing commands (keeping them)
@click.command("get-available-models", help="Gets available models")
@click.option("--api-key", required=True, help="The API key used to query Regolo.")
@click.option(
    "--model-type",
    default="",
    required=False,
    type=click.Choice(
        ["", "chat", "image_generation", "embedding", "audio_transcription", "rerank"]
    ),
    help="The type of the models you want to retrieve (returns all by default)",
)
def get_available_models(api_key: str, model_type: str):
    available_models: list[dict] = regolo.RegoloClient.get_available_models(
        api_key, model_info=True
    )
    output_models: list[tuple] = []
    for model in available_models:
        mode = model.get("model_info", {}).get("mode") or ""
        if not model_type or model_type in mode:
            output_models.append((model["model_name"], model["model_info"]["mode"]))
    click.echo(pprint.pformat(output_models))


@click.command("chat", help="Allows chatting with LLMs")
@click.option(
    "--no-hide",
    required=False,
    is_flag=True,
    default=False,
    help="Do not hide the API key when typing",
)
@click.option("--api-key", required=False, help="The API key used to chat with Regolo.")
@click.option(
    "--disable-newlines",
    required=False,
    is_flag=True,
    default=False,
    help="Disable new lines, they will be replaced with space character",
)
@click.option("--max-tokens", default=2048, help="Max tokens per response (default: 2048)")
def chat(no_hide: bool, api_key: str, disable_newlines: bool, max_tokens: int):
    if not api_key:
        api_key = click.prompt("Insert your regolo API key", hide_input=not no_hide)
    available_models: list[dict] = regolo.RegoloClient.get_available_models(
        api_key, model_info=True
    )

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

    click.echo("\n")

    client = RegoloClient(api_key=api_key, chat_model=model)
    try:
        click.echo(f'You can now chat with {model}, write "/bye" to exit')

        while True:
            # ask user input
            user_input = click.prompt("user")

            if user_input == "/bye":
                exit(0)
            # get chat response and save in the client
            response = client.run_chat(
                user_input, stream=True, full_output=False, max_tokens=max_tokens
            )

            # print output
            was_thinking = False
            while True:
                try:
                    res = next(response)
                    is_thinking = res[0] == "thinking"
                    text = res[1]
                    if is_thinking and not was_thinking:
                        click.echo(click.style("Thinking...", dim=True, italic=True))
                        was_thinking = True
                    elif not is_thinking and was_thinking and text:
                        click.echo(click.style("\n─────────────────────\n", dim=True))
                        was_thinking = False
                    if disable_newlines:
                        text = text.replace("\n", " ")
                    if is_thinking:
                        click.echo(click.style(text, dim=True), nl=False)
                    else:
                        click.echo(text, nl=False)
                except StopIteration:
                    break

            click.echo("\n")
    finally:
        client.close()


@click.command("create-image", help="Creates images")
@click.option("--api-key", required=True, help="The API key used generate.")
@click.option("--model", required=True, help="The number of images to generate. (Defaults to 1)")
@click.option("--save-path", help="The path in which to save the images. (Defaults to ../images)")
@click.option(
    "--prompt",
    default="A generic image",
    help='The text prompt for image generation. (Defaults to "A generic image")',
)
@click.option("--n", default=1, help="The number of images to generate. (Defaults to 1)")
@click.option(
    "--quality",
    default="standard",
    help="The quality of the image that will be generated. The 'hd' value creates images with finer details and greater consistency across the image. (Defaults to 'standard'')",
)
@click.option(
    "--size",
    default="1024x1024",
    help="The size of the generated images. (Defaults to '1024x1024')",
)
@click.option(
    "--style",
    default="realistic",
    help="The style of the generated images. (Defaults to 'realistic')",
)
@click.option(
    "--output-file-format",
    default="png",
    type=click.Choice(["png", "jpg", "jpeg", "webp", "bmp"], case_sensitive=False),
    help="The output file format for the generated images. (Defaults to 'png')",
)
def create_image(
    api_key: str,
    save_path: str | None,
    model: str,
    prompt: str,
    n: int,
    quality: str,
    size: str,
    style: str,
    output_file_format: str,
):
    KeysHandler.check_key(api_key)

    if save_path is None:
        save_path = os.path.join(os.getcwd(), "images")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    client = RegoloClient(api_key=api_key, image_generation_model=model)

    try:
        images_bytes = client.create_image(
            prompt=prompt, n=n, quality=quality, size=size, style=style
        )
    finally:
        client.close()

    # Normalize format extension
    ext = output_file_format.lower()
    if ext == "jpg":
        ext = "jpeg"

    for image_bytes in images_bytes:
        image = Image.open(BytesIO(image_bytes))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(save_path, f"{timestamp}.{ext}")

        i = 1
        while os.path.exists(filepath):
            filename = f"{timestamp}_{i}.{ext}"
            filepath = os.path.join(save_path, filename)
            i += 1

        # Save with the appropriate format
        if ext == "jpeg":
            image.convert("RGB").save(filepath, "JPEG")
        else:
            image.save(filepath, ext.upper())


@click.command("transcribe-audio", help="Transcribes audio files")
@click.option("--api-key", required=True, help="The API key used to transcribe.")
@click.option(
    "--model",
    required=True,
    help="The model to use for transcription (gpt-4o-transcribe, gpt-4o-mini-transcribe, or whisper-1)",
)
@click.option("--file-path", required=True, help="Path to the audio file to transcribe")
@click.option(
    "--save-path", help="Path to save the transcription (prints to console if not specified)"
)
@click.option(
    "--language", help="Language of the input audio in ISO-639-1 format (e.g., en, es, fr)"
)
@click.option(
    "--prompt", help="Optional text to guide the model's style or continue a previous audio segment"
)
@click.option(
    "--response-format",
    default="json",
    type=click.Choice(["json", "text", "srt", "verbose_json", "vtt"]),
    help="Format of the transcript output (defaults to json)",
)
@click.option(
    "--temperature", type=float, help="Sampling temperature between 0 and 1 (defaults to 0)"
)
@click.option("--chunking-strategy", help='Controls audio chunking: "auto" or JSON object string')
@click.option(
    "--include-logprobs",
    is_flag=True,
    help="Include log probabilities (only works with json format and gpt-4o models)",
)
@click.option(
    "--timestamp-granularities",
    multiple=True,
    type=click.Choice(["word", "segment"]),
    help="Timestamp granularities (requires verbose_json format)",
)
@click.option("--stream", is_flag=True, help="Stream the response (not supported for whisper-1)")
@click.option("--full-output", is_flag=True, help="Return full API response instead of just text")
def transcribe_audio(
    api_key: str,
    model: str,
    file_path: str,
    save_path: str,
    language: str,
    prompt: str,
    response_format: str,
    temperature: float,
    chunking_strategy: str,
    include_logprobs: bool,
    timestamp_granularities: tuple,
    stream: bool,
    full_output: bool,
):
    # Validate API key
    KeysHandler.check_key(api_key)

    # Prepare optional parameters
    kwargs: dict[Any, Any] = {
        "file": file_path,
        "model": model,
        "api_key": api_key,
        "response_format": response_format,
        "stream": stream,
        "full_output": full_output,
    }

    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt
    if temperature is not None:
        kwargs["temperature"] = temperature
    if chunking_strategy:
        kwargs["chunking_strategy"] = chunking_strategy
    if include_logprobs:
        kwargs["include"] = ["logprobs"]
    if timestamp_granularities:
        kwargs["timestamp_granularities"] = list(timestamp_granularities)

    try:
        if stream:
            # Handle streaming
            click.echo("Transcribing (streaming)...")
            response = RegoloClient.static_audio_transcription(**kwargs)

            if save_path:
                # Stream to file
                with open(save_path, "w", encoding="utf-8") as f:
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
            response = RegoloClient.static_audio_transcription(**kwargs)
            if not isinstance(response, (str, dict)):
                raise click.ClickException(
                    "Expected text or JSON from a non-streaming transcription"
                )

            # Format output based on response_format and full_output
            if full_output and isinstance(response, dict):
                output = json.dumps(response, indent=2, ensure_ascii=False)
            elif response_format == "json" and isinstance(response, dict):
                output = response.get("text", str(response))
            else:
                output = str(response)

            if save_path:
                # Save to file
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(output)
                click.echo(f"Transcription saved to: {save_path}")
            else:
                # Print to console
                click.echo(output)

    except Exception as e:
        raise click.ClickException(f"Transcription failed: {str(e)}")


@click.command("rerank", help="Rerank documents based on relevance to a query")
@click.option("--api-key", required=True, help="The API key used for reranking.")
@click.option(
    "--model",
    required=True,
    help="The reranking model to use (e.g., jina-reranker-v2, cohere-rerank-v3)",
)
@click.option("--query", required=True, help="The search query to compare documents against")
@click.option(
    "--documents", multiple=True, help="Documents to rerank (can be specified multiple times)"
)
@click.option("--documents-file", help="Path to JSON file containing documents array")
@click.option(
    "--top-n",
    type=int,
    help="Number of most relevant documents to return (returns all if not specified)",
)
@click.option(
    "--rank-fields",
    multiple=True,
    help="For structured documents, specify which fields to rank by (can be specified multiple times)",
)
@click.option(
    "--no-return-documents",
    is_flag=True,
    default=False,
    help="Do not return document content in results (only indices and scores)",
)
@click.option("--max-chunks-per-doc", type=int, help="Maximum number of chunks per document")
@click.option(
    "--save-path",
    help="Path to save the reranking results as JSON (prints to console if not specified)",
)
@click.option(
    "--full-output", is_flag=True, help="Return full API response instead of just results"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"]),
    default="table",
    help="Output format (defaults to table)",
)
def rerank_documents(
    api_key: str,
    model: str,
    query: str,
    documents: tuple,
    documents_file: str,
    top_n: int,
    rank_fields: tuple,
    no_return_documents: bool,
    max_chunks_per_doc: int,
    save_path: str,
    full_output: bool,
    output_format: str,
):
    """Rerank documents based on their relevance to a query"""

    # Validate API key
    KeysHandler.check_key(api_key)

    # Prepare documents list
    docs_list = []

    if documents_file and documents:
        raise click.ClickException("Use either --documents or --documents-file")
    if documents_file:
        # Load documents from JSON file
        try:
            with open(documents_file, "r", encoding="utf-8") as f:
                docs_list = json.load(f)

            if not isinstance(docs_list, list):
                raise click.ClickException("Documents file must contain a JSON array")

        except FileNotFoundError:
            raise click.ClickException(f"Documents file not found: {documents_file}")
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in documents file: {e}")
    else:
        # Use documents from the command line
        if not documents:
            raise click.ClickException("Either --documents or --documents-file must be specified")
        docs_list = list(documents)

    if not docs_list:
        raise click.ClickException("No documents provided for reranking")

    # Prepare optional parameters
    kwargs: dict[str, Any] = {
        "query": query,
        "documents": docs_list,
        "return_documents": not no_return_documents,
        "full_output": full_output,
    }

    if top_n is not None:
        kwargs["top_n"] = top_n
    if rank_fields:
        kwargs["rank_fields"] = list(rank_fields)
    if max_chunks_per_doc is not None:
        kwargs["max_chunks_per_doc"] = max_chunks_per_doc

    # Create a client and rerank
    client = RegoloClient(api_key=api_key, reranker_model=model)

    try:
        click.echo(f"Reranking {len(docs_list)} documents with query: '{query}'...")

        try:
            response = client.rerank(**kwargs)
        except Exception as e:
            print(e)
            raise click.ClickException(f"Reranking failed: {str(e)}")
        # Format output
        if output_format == "json":
            output = json.dumps(response, indent=2, ensure_ascii=False)
        else:
            # Table format
            if isinstance(response, dict):
                results: list[dict[str, Any]] = response.get("results", [])
                metadata = {k: v for k, v in response.items() if k != "results"}

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
                index = result.get("index", "N/A")
                score = result.get("relevance_score", 0.0)

                # Format score as percentage
                score_pct = f"{score * 100:.2f}%" if isinstance(score, (int, float)) else str(score)

                output_lines.append(f"  {i}. Document #{index} - Relevance: {score_pct}")

                if "document" in result:
                    doc = result["document"]
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
            with open(save_path, "w", encoding="utf-8") as f:
                if output_format == "json":
                    f.write(output)
                else:
                    # Save as JSON even if displayed as a table
                    f.write(json.dumps(response, indent=2, ensure_ascii=False))
            click.echo(f"✅ Reranking results saved to: {save_path}")
        else:
            click.echo(output)

    except Exception as e:
        raise click.ClickException(f"Reranking failed: {str(e)}")
    finally:
        client.close()


@auth.command("refresh")
def refresh_auth():
    """Refresh saved credentials without printing tokens."""
    try:
        result = _get_model_client().refresh_access_token()
        click.echo(
            f"Credentials refreshed; expires in {result.get('expires_in', 'unknown')} seconds"
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@models.command("retry-scan")
@click.argument("model_name")
def retry_scan(model_name):
    """Retry a scan when server policy permits it."""
    try:
        click.echo(json.dumps(_get_model_client().retry_model_scan(model_name), indent=2))
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@models.command("wait")
@click.argument("model_name")
@click.option("--timeout", type=click.FloatRange(min=0), default=3600, show_default=True)
def wait_scan(model_name, timeout):
    """Wait for a safe scan, stopping on actionable security/capacity blockers."""
    try:
        click.echo(
            json.dumps(_get_model_client().wait_for_scan(model_name, timeout=timeout), indent=2)
        )
    except (httpx.HTTPError, RuntimeError, ValueError, TimeoutError) as exc:
        raise click.ClickException(str(exc)) from exc


def history_options(function):
    for option in reversed(
        [
            click.option("--month", help="MMYYYY; UTC month, mutually exclusive with a range"),
            click.option("--time-range-start", help="ISO 8601 start with timezone"),
            click.option("--time-range-end", help="Exclusive ISO 8601 end with timezone"),
            click.option("--model-name", help="Filter by registered model name"),
            click.option("--status", help="Stored status, e.g. error (runtime uses failed)"),
        ]
    ):
        function = option(function)
    return function


@inference.command("sessions")
@history_options
@click.option("--limit", type=click.IntRange(1, 200), default=50, show_default=True)
@click.option("--before-session-id", type=click.IntRange(min=1))
@click.option("--include-status", is_flag=True)
@click.option("--include-error", is_flag=True, help="Include full, potentially large failure text")
@click.option("--include-billing", is_flag=True)
@click.option("--include-scaling", is_flag=True)
@click.option("--all", "all_pages", is_flag=True, help="Follow all cursors")
@click.option(
    "--csv", "csv_path", type=click.Path(dir_okay=False), help="Export all matching sessions"
)
@click.option(
    "--json-file",
    type=click.Path(dir_okay=False),
    help="Export all pages and nested fields as JSON",
)
def sessions(all_pages, csv_path, json_file=None, **filters):
    """Discover pending, running, completed, stopped and failed sessions."""
    try:
        if csv_path and json_file:
            raise ValueError("Choose one export format: --csv or --json-file")
        if json_file:
            count = _get_model_client().export_sessions_json(json_file, **filters)
            click.echo(f"Exported {count} sessions to {json_file}")
        elif csv_path:
            filters["include_status"] = True
            filters["include_billing"] = True
            count = _get_model_client().export_sessions_csv(csv_path, **filters)
            click.echo(f"Exported {count} sessions to {csv_path}")
        elif all_pages:
            click.echo(
                json.dumps(list(_get_model_client().iter_inference_sessions(**filters)), indent=2)
            )
        else:
            click.echo(json.dumps(_get_model_client().get_inference_sessions(**filters), indent=2))
    except (httpx.HTTPError, RuntimeError, ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


@inference.command("usage")
@history_options
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table")
def usage(output_format, **filters):
    """Aggregate server billing across all history pages, grouped by model/GPU."""
    try:
        result = _get_model_client().get_usage_summary(**filters)
        if output_format == "json":
            click.echo(json.dumps(result, indent=2))
            return
        click.echo(f"Sessions: {result['sessions']} ({result['open_sessions']} open)")
        click.echo(f"Cost: €{result['cost_euro']:.2f}")
        click.echo(f"Wall-clock seconds: {result['duration_seconds']:g}")
        click.echo(f"Replica seconds: {result['replica_seconds']:g}")
        click.echo(f"Billable replica hours: {result['billable_replica_hours']:g}")
        for label in ("by_model", "by_gpu"):
            click.echo(f"\n{label.replace('_', ' ').title()}:")
            for name, item in result[label].items():
                click.echo(f"  {name}: {item['sessions']} sessions, €{item['cost_euro']:.2f}")
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@inference.command("session-status")
@click.argument("session_id", type=click.IntRange(min=1))
@click.option("--wait", "wait_ready", is_flag=True)
@click.option("--timeout", type=click.FloatRange(min=0), default=1800, show_default=True)
def session_status(session_id, wait_ready, timeout):
    """Read one session, including its final failure or administrative stop."""
    try:
        result = (
            _get_model_client().wait_for_inference(session_id, timeout=timeout)
            if wait_ready
            else _get_model_client().get_runtime_status(session_id)
        )
        click.echo(json.dumps(result, indent=2))
    except (httpx.HTTPError, RuntimeError, ValueError, TimeoutError) as exc:
        raise click.ClickException(str(exc)) from exc


@inference.command("billing")
@click.argument("session_id", type=click.IntRange(min=1))
def session_billing(session_id):
    """Show lifetime server-calculated billing for one session."""
    try:
        click.echo(json.dumps(_get_model_client().get_runtime_billing(session_id), indent=2))
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@inference.command("details")
@click.argument("session_id", type=click.IntRange(min=1))
@click.option("--include-scaling", is_flag=True)
@click.option("--include-error", is_flag=True)
def session_details(session_id, include_scaling, include_error):
    """Full owned session history, runtime status and billing fields."""
    try:
        click.echo(
            json.dumps(
                _get_model_client().get_session_details(
                    session_id, include_scaling=include_scaling, include_error=include_error
                ),
                indent=2,
            )
        )
    except (httpx.HTTPError, RuntimeError, ValueError, LookupError) as exc:
        raise click.ClickException(str(exc)) from exc


@inference.command("scaling")
@click.argument("session_id", type=click.IntRange(min=1))
@click.option("--limit", type=click.IntRange(min=1), default=100)
@click.option("--cursor", help="Cursor returned by the previous server page")
def session_scaling(session_id, limit, cursor):
    """One scaling page, using the route in the server's public OpenAPI."""
    try:
        click.echo(
            json.dumps(
                _get_model_client().get_scaling_history(session_id, limit=limit, cursor=cursor),
                indent=2,
            )
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@inference.command("overview")
@click.option("--month", help="Usage month MMYYYY; default current UTC month")
@click.option("--include-scaling", is_flag=True)
def dashboard_overview(month, include_scaling):
    """Complete user model/GPU/deployment/usage snapshot as JSON."""
    try:
        click.echo(
            json.dumps(
                _get_model_client().get_dashboard_snapshot(
                    month=month, include_scaling=include_scaling
                ),
                indent=2,
            )
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@cli.command("catalog")
@click.option("--api-key", envvar="REGOLO_API_KEY", required=True)
@click.option("--base-url", default="https://api.regolo.ai", show_default=True)
@click.option(
    "--mode", help="chat, embedding, rerank, image_generation, ocr or audio_transcription"
)
@click.option("--supports", multiple=True, help="Required capability, e.g. vision or reasoning")
def model_catalog(api_key, base_url, mode, supports):
    """Catalog capabilities, prices, limits and supported request parameters."""
    try:
        with RegoloClient(api_key=api_key, alternative_url=base_url) as client:
            click.echo(
                json.dumps(client.get_model_catalog(mode=mode, supports=list(supports)), indent=2)
            )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


inference.add_command(inference.commands["status"], "loaded")

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

if __name__ == "__main__":
    cli()
