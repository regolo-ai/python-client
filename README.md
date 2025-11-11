# **Regolo.ai Python Client**

A comprehensive Python client for interacting with **Regolo.ai's** LLM-based API and Model Management platform.

## **Table of Contents**

- [Installation](#installation)
- [Basic Usage](#basic-usage)
  - [Chat and Completions](#chat-and-completions)
  - [Image Generation](#image-generation)
  - [Audio Transcription](#audio-transcription)
  - [Text Embeddings](#text-embeddings)
  - [Document Reranking](#document-reranking)
- [CLI Usage](#cli-usage)
  - [Chat Interface](#chat-interface)
  - [Model Management](#model-management)
  - [Inference Management](#inference-management)
  - [SSH Key Management](#ssh-key-management)
- [Model Management & Deployment](#model-management--deployment)
  - [Authentication](#authentication)
  - [Registering Models](#registering-models)
  - [Loading Models for Inference](#loading-models-for-inference)
  - [Monitoring and Billing](#monitoring-and-billing)
- [Advanced Usage](#advanced-usage)
- [Environment Variables](#environment-variables)

---

## **Installation**

Install the `regolo` package using pip:

```bash
pip install regolo
```

---

## **Basic Usage**

### **1. Import the regolo module**

```python
import regolo
```

### **2. Set Up Default API Key and Model**

To avoid manually passing the API key and model in every request, you can set them globally:

```python
regolo.default_key = "<YOUR_API_KEY>"
regolo.default_chat_model = "Llama-3.3-70B-Instruct"
```

This ensures that all `RegoloClient` instances and static functions will use the specified API key and model.

You can still override these defaults by passing parameters directly to methods.

### **Chat and Completions**

#### **Text Completion:**
```python
# Using static method
response = regolo.static_completions(prompt="Tell me something about Rome.")
print(response)

# Using client instance
client = regolo.RegoloClient()
response = client.completions(prompt="Tell me something about Rome.")
print(response)
```

#### **Chat Completion:**
```python
# Using static method
role, content = regolo.static_chat_completions(
    messages=[{"role": "user", "content": "Tell me something about Rome"}]
)
print(f"{role}: {content}")

# Using client instance
client = regolo.RegoloClient()
role, content = client.run_chat(user_prompt="Tell me something about Rome")
print(f"{role}: {content}")
```

#### **Handling Conversation History:**
```python
client = regolo.RegoloClient()

# Add prompts to conversation
client.add_prompt_to_chat(role="user", prompt="Tell me about Rome!")
print(client.run_chat())

# Continue the conversation
client.add_prompt_to_chat(role="user", prompt="Tell me more about its history!")
print(client.run_chat())

# View full conversation
print(client.instance.get_conversation())

# Clear conversation to start fresh
client.clear_conversations()
```

#### **Streaming Responses:**

**With full output:**
```python
client = regolo.RegoloClient()
response = client.run_chat(
    user_prompt="Tell me about Rome",
    stream=True,
    full_output=True
)

while True:
    try:
        print(next(response))
    except StopIteration:
        break
```

**Without full output (text only):**
```python
client = regolo.RegoloClient()
response = client.run_chat(
    user_prompt="Tell me about Rome",
    stream=True,
    full_output=False
)

while True:
    try:
        role, content = next(response)
        if role:
            print(f"{role}:")
        print(content, end="", flush=True)
    except StopIteration:
        break
```

### **Image Generation**

**Without client:**
```python
from io import BytesIO
from PIL import Image
import regolo

regolo.default_image_generation_model = "Qwen-Image"
regolo.default_key = "<YOUR_API_KEY>"

img_bytes = regolo.static_image_create(prompt="a cat")[0]
image = Image.open(BytesIO(img_bytes))
image.show()
```

**With client:**
```python
from io import BytesIO
from PIL import Image
import regolo

client = regolo.RegoloClient(
    image_generation_model="Qwen-Image",
    api_key="<YOUR_API_KEY>"
)

img_bytes = client.create_image(prompt="A cat in Rome")[0]
image = Image.open(BytesIO(img_bytes))
image.show()
```

**Generate multiple images:**
```python
client = regolo.RegoloClient()
images = client.create_image(
    prompt="Beautiful landscape",
    n=3,  # Generate 3 images
    quality="hd",
    size="1024x1024",
    style="realistic"
)

for i, img_bytes in enumerate(images):
    image = Image.open(BytesIO(img_bytes))
    image.save(f"output_{i}.png")
```

### **Audio Transcription**

**Without client:**
```python
import regolo

regolo.default_key = "<YOUR_API_KEY>"
regolo.default_audio_transcription_model = "faster-whisper-large-v3"

transcribed_text = regolo.static_audio_transcription(
    file="path/to/audio.mp3",
    full_output=True
)
print(transcribed_text)
```

**With client:**
```python
import regolo

client = regolo.RegoloClient(
    api_key="<YOUR_API_KEY>",
    audio_transcription_model="faster-whisper-large-v3"
)

transcribed_text = client.audio_transcription(
    file="path/to/audio.mp3",
    language="en",  # Optional: specify language
    response_format="json"  # Options: json, text, srt, verbose_json, vtt
)
print(transcribed_text)
```

**Streaming transcription:**
```python
client = regolo.RegoloClient()
response = client.audio_transcription(
    file="path/to/audio.mp3",
    stream=True
)

for chunk in response:
    print(chunk, end="", flush=True)
```

### **Text Embeddings**

**Without client:**
```python
import regolo

regolo.default_key = "<YOUR_API_KEY>"
regolo.default_embedder_model = "gte-Qwen2"

embeddings = regolo.static_embeddings(input_text=["test", "test1"])
print(embeddings)
```

**With client:**
```python
import regolo

client = regolo.RegoloClient(
    api_key="<YOUR_API_KEY>",
    embedder_model="gte-Qwen2"
)

# Single text
embedding = client.embeddings(input_text="Hello world")
print(embedding)

# Multiple texts
embeddings = client.embeddings(input_text=["text1", "text2", "text3"])
for i, emb in enumerate(embeddings):
    print(f"Embedding {i}: {emb['embedding'][:5]}...")  # First 5 dimensions
```

### **Document Reranking**

**Without client:**
```python
import regolo

regolo.default_key = "<YOUR_API_KEY>"
regolo.default_reranker_model = "jina-reranker-v2"

documents = [
    "Paris is the capital of France",
    "Berlin is the capital of Germany",
    "Rome is the capital of Italy"
]

results = regolo.RegoloClient.static_rerank(
    query="What is the capital of France?",
    documents=documents,
    api_key=regolo.default_key,
    model=regolo.default_reranker_model,
    top_n=2
)

for result in results:
    print(f"Document {result['index']}: {result['relevance_score']:.4f}")
    if 'document' in result:
        print(f"  Content: {result['document']}")
```

**With client:**
```python
import regolo

client = regolo.RegoloClient(
    api_key="<YOUR_API_KEY>",
    reranker_model="jina-reranker-v2"
)

documents = [
    {"title": "Doc1", "text": "Paris is the capital of France"},
    {"title": "Doc2", "text": "Berlin is the capital of Germany"}
]

results = client.rerank(
    query="French capital",
    documents=documents,
    top_n=1,
    rank_fields=["text"],  # For structured documents
    return_documents=True
)

print(f"Most relevant: {results[0]['document']}")
```

---

## **CLI Usage**

The Regolo CLI provides a comprehensive interface for model management, inference deployment, and API interactions.

### **Chat Interface**

Start an interactive chat session:

```bash
regolo chat
```

**Options:**
- `--no-hide`: Display API key while typing
- `--disable-newlines`: Replace newlines with spaces in responses
- `--api-key <key>`: Provide API key directly instead of being prompted

**Example:**
```bash
regolo chat --api-key <YOUR_API_KEY> --disable-newlines
```

### **Model Management**

#### **Authentication**

Before using model management features, authenticate with your credentials:

```bash
regolo auth login
```

You'll be prompted for your username and password. The CLI will save your authentication tokens automatically.

**Logout:**
```bash
regolo auth logout
```

#### **List Available Models**

Get models accessible with your API key:

```bash
regolo get-available-models --api-key <YOUR_API_KEY>
```

Filter by model type:
```bash
regolo get-available-models --api-key <YOUR_API_KEY> --model-type chat
# Options: chat, image_generation, embedding, audio_transcription, rerank
```

#### **Register Models**

**Register a HuggingFace model:**
```bash
regolo models register \
  --name my-llama-model \
  --type huggingface \
  --url meta-llama/Llama-2-7b-hf \
  --api-key <HF_TOKEN>  # Optional, for private models
```

**Register a custom model:**
```bash
regolo models register \
  --name my-custom-model \
  --type custom
```

This creates a GitLab repository at `git@gitlab.regolo.ai:<username>/my-custom-model.git`

#### **List Registered Models**

```bash
regolo models list
```

**JSON output:**
```bash
regolo models list --format json
```

#### **Get Model Details**

```bash
regolo models details my-llama-model
```

#### **Delete a Model**

```bash
regolo models delete my-llama-model --confirm
```

### **Inference Management**

#### **View Available GPUs**

```bash
regolo inference gpus
```

**JSON output:**
```bash
regolo inference gpus --format json
```

#### **Load Model for Inference**

**Interactive (will prompt for GPU selection):**
```bash
regolo inference load my-llama-model
```

**With specific GPU:**
```bash
regolo inference load my-llama-model --gpu required-gpu
```

**With vLLM configuration:**
```bash
regolo inference load my-llama-model \
  --gpu ECS1GPU11 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1
```

**Using vLLM config file:**
```bash
# Create vllm_config.json
cat > vllm_config.json << EOF
{
  "max_model_len": 4096,
  "gpu_memory_utilization": 0.9,
  "tensor_parallel_size": 1,
  "disable_log_requests": true
}
EOF

regolo inference load my-llama-model \
  --gpu ECS1GPU11 \
  --vllm-config-file vllm_config.json
```

**Force overwrite existing configuration:**
```bash
regolo inference load my-llama-model --gpu ECS1GPU11 --force
```

#### **View Loaded Models**

```bash
regolo inference status
```

This shows:
- Session IDs
- Model names
- GPU assignments
- Load times
- Current costs

#### **Unload Model**

**Interactive (will show loaded models):**
```bash
regolo inference unload
```

**By session ID:**
```bash
regolo inference unload --session-id 12345
```

**By model name:**
```bash
regolo inference unload --model-name my-llama-model
```

#### **Monitor Costs**

**Current month:**
```bash
regolo inference user-status
```

**Specific month (MMYYYY format):**
```bash
regolo inference user-status --month 012025
```

**Time range:**
```bash
regolo inference user-status \
  --time-range-start 2025-01-01T00:00:00Z \
  --time-range-end 2025-01-15T23:59:59Z
```

**JSON output:**
```bash
regolo inference user-status --format json
```

### **SSH Key Management**

SSH keys are required to push custom model files to GitLab repositories.

#### **Add SSH Key**

**From file:**
```bash
regolo ssh add \
  --title "My Development Key" \
  --key-file ~/.ssh/id_rsa.pub
```

**Direct key content:**
```bash
regolo ssh add \
  --title "My Development Key" \
  --key "ssh-rsa AAAAB3NzaC1yc2E... user@example.com"
```

#### **List SSH Keys**

```bash
regolo ssh list
```

**JSON output:**
```bash
regolo ssh list --format json
```

#### **Delete SSH Key**

```bash
regolo ssh delete <KEY_ID> --confirm
```

### **Complete Workflow Command**

The workflow command automates the entire model deployment process:

```bash
regolo workflow workflow my-custom-model \
  --type custom \
  --ssh-key-file ~/.ssh/id_rsa.pub \
  --ssh-key-title "Dev Key" \
  --local-model-path ./my_model_files \
  --auto-load
```

This will:
1. Register the model
2. Add your SSH key
3. Guide you through uploading files to GitLab
4. Automatically load the model for inference (if `--auto-load`)

**For HuggingFace models:**
```bash
regolo workflow workflow my-gpt2 \
  --type huggingface \
  --url gpt2 \
  --auto-load
```

### **Other CLI Commands**

#### **Create Images**

```bash
regolo create-image \
  --api-key <YOUR_API_KEY> \
  --model Qwen-Image \
  --prompt "A beautiful sunset" \
  --n 2 \
  --size 1024x1024 \
  --quality hd \
  --style realistic \
  --save-path ./images \
  --output-file-format png
```

#### **Transcribe Audio**

```bash
regolo transcribe-audio \
  --api-key <YOUR_API_KEY> \
  --model faster-whisper-large-v3 \
  --file-path audio.mp3 \
  --language en \
  --response-format json \
  --save-path transcription.txt
```

**Streaming transcription:**
```bash
regolo transcribe-audio \
  --api-key <YOUR_API_KEY> \
  --model faster-whisper-large-v3 \
  --file-path audio.mp3 \
  --stream
```

#### **Rerank Documents**

```bash
regolo rerank \
  --api-key <YOUR_API_KEY> \
  --model jina-reranker-v2 \
  --query "capital of France" \
  --documents "Paris is the capital" \
  --documents "Berlin is the capital" \
  --documents "Rome is the capital" \
  --top-n 2
```

**Using a documents file:**
```bash
# Create documents.json
cat > documents.json << EOF
[
  "Paris is the capital of France",
  "Berlin is the capital of Germany",
  "Rome is the capital of Italy"
]
EOF

regolo rerank \
  --api-key <YOUR_API_KEY> \
  --model jina-reranker-v2 \
  --query "capital of France" \
  --documents-file documents.json \
  --format table
```

---

## **Model Management & Deployment**

The Regolo platform provides comprehensive model management capabilities, allowing you to register,
deploy, and monitor both HuggingFace and custom models on GPU infrastructure.

### **Authentication**

Before using model management features, authenticate using the CLI:

```bash
regolo auth login
```

Or in Python:

```python
from regolo.cli import ModelManagementClient

client = ModelManagementClient(base_url="https://devmid.regolo.ai")
auth_response = client.authenticate("username", "password")
print(f"Token expires in {auth_response['expires_in']} seconds")

# Save tokens for future use
access_token = auth_response['access_token']
refresh_token = auth_response['refresh_token']
```

### **Registering Models**

#### **HuggingFace Models**

Register a model from HuggingFace Hub:

```bash
regolo models register \
  --name my-bert-model \
  --type huggingface \
  --url bert-base-uncased
```

For private models, include your HuggingFace token:

```bash
regolo models register \
  --name my-private-model \
  --type huggingface \
  --url organization/private-model \
  --api-key hf_xxxxxxxxxxxxx
```

**Supported URL formats:**
- Full URL: `https://huggingface.co/bert-base-uncased`
- Short format: `bert-base-uncased`
- Organization format: `BAAI/bge-small-en-v1.5`

#### **Custom Models**

For custom models, the platform creates a GitLab repository where you can push your model files:

```bash
# 1. Register the model
regolo models register \
  --name my-custom-model \
  --type custom

# 2. Add SSH key for repository access
regolo ssh add \
  --title "Development Key" \
  --key-file ~/.ssh/id_rsa.pub

# 3. Clone the repository
git clone git@gitlab.regolo.ai:<username>/my-custom-model.git
cd my-custom-model

# 4. Add your model files
# Directory structure example:
# my-custom-model/
# ├── config.json
# ├── tokenizer.json
# ├── tokenizer_config.json
# ├── special_tokens_map.json
# ├── pytorch_model.bin (or model.safetensors)
# └── vocab.txt

cp -r /path/to/your/model/* .

# 5. Commit and push
git add .
git commit -m "Add model files"
git push origin main
```

**Using Git LFS for large files:**

```bash
# Initialize Git LFS
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
git commit -m "Configure Git LFS"
git push origin main
```

**In Python:**

```python
from regolo.cli import ModelManagementClient

client = ModelManagementClient()
client.authenticate("username", "password")

# Register HuggingFace model
hf_result = client.register_model(
    name="my-gpt2",
    is_huggingface=True,
    url="gpt2"
)

# Register custom model
custom_result = client.register_model(
    name="my-custom-llm",
    is_huggingface=False
)

# Add SSH key
ssh_result = client.add_ssh_key(
    title="Dev Key",
    key="ssh-rsa AAAAB3NzaC1yc2E... user@example.com"
)
```

### **Loading Models for Inference**

Once registered, load models onto GPU infrastructure for inference:

```bash
# View available GPUs
regolo inference gpus

# Load model with specific configuration
regolo inference load my-bert-model \
  --gpu ECS1GPU11 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1
```

**vLLM Configuration Options:**

- `--max-model-len`: Maximum sequence length
- `--gpu-memory-utilization`: GPU memory fraction (0.0-1.0)
- `--tensor-parallel-size`: Number of GPUs for tensor parallelism
- `--disable-log-requests`: Disable request logging
- `--enable-auto-tool-choice`: Enable automatic tool choice
- `--tool-call-parser`: Tool call parser (e.g., llama3_json)
- `--chat-template`: Path to chat template file

**In Python:**

```python
from regolo.cli import ModelManagementClient

client = ModelManagementClient()
client.authenticate("username", "password")

# Get available GPUs
gpus = client.get_available_gpus()
gpu_instance = gpus['gpus'][0]['InstanceType']

# Load model with vLLM configuration
vllm_config = {
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9,
    "tensor_parallel_size": 1
}

result = client.load_model_for_inference(
    model_name="my-bert-model",
    gpu=gpu_instance,
    vllm_config=vllm_config
)
```

### **Monitoring and Billing**

#### **View Loaded Models**

```bash
regolo inference status
```

Output includes:
- Session ID (required for unloading)
- Model name
- GPU assignment
- Load time
- Current cost

#### **Monitor Costs**

```bash
# Current month
regolo inference user-status

# Specific month (MMYYYY format)
regolo inference user-status --month 012025

# Custom time range
regolo inference user-status \
  --time-range-start 2025-01-01T00:00:00Z \
  --time-range-end 2025-01-15T23:59:59Z
```

**Billing Details:**
- Hourly billing, rounded up to next full hour
- Minimum charge: 1 hour
- Cost = duration_hours × hourly_price (in EUR)

**In Python:**

```python
# Get loaded models
loaded = client.get_loaded_models()
for model in loaded['loaded_models']:
    print(f"Model: {model['model_name']}")
    print(f"Session: {model['session_id']}")
    print(f"Cost: €{model['cost']:.2f}")

# Get cost status
status = client.get_user_inference_status()
print(f"Total sessions: {status['total']}")
print(f"Total cost: €{status.get('total_cost', 0):.2f}")

# Generate monthly report
status = client.get_user_inference_status(month="012025")
for inference in status['inferences']:
    print(f"{inference['model_name']}: "
          f"{inference['duration_hours']}h, "
          f"€{inference['cost_euro']:.2f}")
```

#### **Unload Models**

Stop billing by unloading models when not in use:

```bash
# Interactive (shows loaded models)
regolo inference unload

# By session ID
regolo inference unload --session-id 12345

# By model name
regolo inference unload --model-name my-bert-model
```

**In Python:**

```python
# Get loaded models
loaded = client.get_loaded_models()

# Unload specific model
for model in loaded['loaded_models']:
    if model['model_name'] == 'my-bert-model':
        client.unload_model_from_inference(model['session_id'])
        break

# Unload all models
for model in loaded['loaded_models']:
    client.unload_model_from_inference(model['session_id'])
```

### **Complete Workflow Example**

```bash
# 1. Authenticate
regolo auth login

# 2. Register HuggingFace model
regolo models register \
  --name llama-2-7b \
  --type huggingface \
  --url meta-llama/Llama-2-7b-hf

# 3. View available GPUs
regolo inference gpus

# 4. Load model for inference
regolo inference load llama-2-7b \
  --gpu ECS1GPU11 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9

# 5. Monitor status (wait for loading to complete)
regolo inference status

# 6. Use the model via API
# (Model is now available through Regolo inference endpoints)

# 7. Check costs
regolo inference user-status

# 8. Unload when done
regolo inference unload --model-name llama-2-7b

# 9. Logout
regolo auth logout
```

**Python equivalent:**

```python
from regolo.cli import ModelManagementClient
import time

# Initialize and authenticate
client = ModelManagementClient()
client.authenticate("username", "password")

# Register model
client.register_model(
    name="llama-2-7b",
    is_huggingface=True,
    url="meta-llama/Llama-2-7b-hf"
)

# Get GPU and load model
gpus = client.get_available_gpus()
gpu = gpus['gpus'][0]['InstanceType']

vllm_config = {
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9
}

client.load_model_for_inference(
    model_name="llama-2-7b",
    gpu=gpu,
    vllm_config=vllm_config
)

# Wait for model to load
print("Waiting for model to load...")
time.sleep(60)  # Adjust based on model size

# Check status
loaded = client.get_loaded_models()
print(f"Loaded models: {loaded['total']}")

# Use model through API...

# Monitor costs
status = client.get_user_inference_status()
print(f"Current cost: €{status.get('total_cost', 0):.2f}")

# Unload when done
for model in loaded['loaded_models']:
    if model['model_name'] == 'llama-2-7b':
        client.unload_model_from_inference(model['session_id'])
```

---

## **Advanced Usage**

### **Switching Models**

```python
client = regolo.RegoloClient(chat_model="Llama-3.3-70B-Instruct")

# Use the model
response = client.run_chat(user_prompt="Hello")

# Switch to a different model
client.change_model("gpt-4o")

# Now using the new model
response = client.run_chat(user_prompt="Hello again")
```

### **Managing Conversation State**

```python
from regolo.instance.structures.conversation_model import Conversation, ConversationLine

client = regolo.RegoloClient()

# Build conversation manually
conversation = Conversation(lines=[
    ConversationLine(role="user", content="What is Python?"),
    ConversationLine(role="assistant", content="Python is a programming language."),
    ConversationLine(role="user", content="Tell me more.")
])

# Use existing conversation
client.instance.overwrite_conversation(conversation)
response = client.run_chat()

# Or create a new client from an existing instance
new_client = regolo.RegoloClient.from_instance(client.instance)
```

### **Custom Base URL**

```python
# Use a custom Regolo server
client = regolo.RegoloClient(
    api_key="<YOUR_API_KEY>",
    alternative_url="https://custom.regolo-server.com"
)
```

### **Reusing HTTP Client**

```python
import httpx

# Create a persistent HTTP client
http_client = httpx.Client()

# Reuse across multiple RegoloClient instances
client1 = regolo.RegoloClient(pre_existent_client=http_client)
client2 = regolo.RegoloClient(pre_existent_client=http_client)

# Don't forget to close when done
http_client.close()
```

### **Working with Full Response Objects**

```python
client = regolo.RegoloClient()

# Get full API response
response = client.run_chat(
    user_prompt="Hello",
    full_output=True
)

print(response)  # Full API response dict
# {
#   "choices": [...],
#   "usage": {...},
#   "model": "...",
#   ...
# }
```

---

## **Environment Variables**

### **Default Values**

Configure default settings via environment variables:

#### **API_KEY**
Set your default API key:
```bash
export API_KEY="<YOUR_API_KEY>"
```

Load in Python:
```python
import regolo
regolo.key_load_from_env_if_exists()
# Now regolo.default_key is set
```

#### **LLM**
Set your default chat model:
```bash
export LLM="Llama-3.3-70B-Instruct"
```

Load in Python:
```python
import regolo
regolo.default_chat_model_load_from_env_if_exists()
# Now regolo.default_chat_model is set
```

#### **IMAGE_MODEL**
Set your default image generation model:
```bash
export IMAGE_MODEL="Qwen-Image"
```

Load in Python:
```python
import regolo
regolo.default_image_load_from_env_if_exists()
# Now regolo.default_image_generation_model is set
```

#### **EMBEDDER_MODEL**
Set your default embedder model:
```bash
export EMBEDDER_MODEL="gte-Qwen2"
```

Load in Python:
```python
import regolo
regolo.default_embedder_load_from_env_if_exists()
# Now regolo.default_embedder_model is set
```

#### **Load All Defaults**
Load all default environment variables at once:
```python
import regolo
regolo.try_loading_from_env()
```

### **Endpoints**

Configure API endpoints (usually not needed):

#### **REGOLO_URL**
```bash
export REGOLO_URL="https://api.regolo.ai"
```

#### **COMPLETIONS_URL_PATH**
```bash
export COMPLETIONS_URL_PATH="/v1/completions"
```

#### **CHAT_COMPLETIONS_URL_PATH**
```bash
export CHAT_COMPLETIONS_URL_PATH="/v1/chat/completions"
```

#### **IMAGE_GENERATION_URL_PATH**
```bash
export IMAGE_GENERATION_URL_PATH="/v1/images/generations"
```

#### **EMBEDDINGS_URL_PATH**
```bash
export EMBEDDINGS_URL_PATH="/v1/embeddings"
```

#### **AUDIO_TRANSCRIPTION_URL_PATH**
```bash
export AUDIO_TRANSCRIPTION_URL_PATH="/v1/audio/transcriptions"
```

#### **RERANK_URL_PATH**
```bash
export RERANK_URL_PATH="/v1/rerank"
```

> [!TIP]
> Endpoint environment variables can be changed during execution since the client works directly with them.
> However, you typically won't need to change these as they're tied to the official Regolo API structure.

---

## **Complete Examples**

### **Multi-Model Workflow**

```python
import regolo
from io import BytesIO
from PIL import Image

# Configure defaults
regolo.default_key = "<YOUR_API_KEY>"
regolo.default_chat_model = "Llama-3.3-70B-Instruct"
regolo.default_image_generation_model = "Qwen-Image"
regolo.default_embedder_model = "gte-Qwen2"

# 1. Chat about a topic
client = regolo.RegoloClient()
response = client.run_chat(user_prompt="Describe a futuristic city")
description = response[1] if isinstance(response, tuple) else response

print(f"Description: {description}")

# 2. Generate image based on description
img_client = regolo.RegoloClient()
img_bytes = img_client.create_image(
    prompt=description[:500],  # Use first 500 chars
    n=1,
    quality="hd"
)[0]

# Save image
image = Image.open(BytesIO(img_bytes))
image.save("futuristic_city.png")
print("Image saved!")

# 3. Create embeddings for search
emb_client = regolo.RegoloClient()
texts = [
    "futuristic city with flying cars",
    "modern urban landscape",
    "ancient historical architecture"
]
embeddings = emb_client.embeddings(input_text=texts)
print(f"Generated {len(embeddings)} embeddings")

# 4. Rerank documents by relevance
rerank_client = regolo.RegoloClient(reranker_model="jina-reranker-v2")
results = rerank_client.rerank(
    query="futuristic technology",
    documents=texts,
    top_n=2
)

print("\nMost relevant documents:")
for result in results:
    print(f"  {result['relevance_score']:.4f}: {texts[result['index']]}")
```

### **Batch Processing**

```python
import regolo

regolo.default_key = "<YOUR_API_KEY>"
regolo.default_chat_model = "Llama-3.3-70B-Instruct"

client = regolo.RegoloClient()

# Process multiple prompts
prompts = [
    "Summarize machine learning in one sentence",
    "Explain quantum computing briefly",
    "What is blockchain technology?"
]

responses = []
for prompt in prompts:
    role, content = client.run_chat(user_prompt=prompt)
    responses.append(content)
    client.clear_conversations()  # Start fresh for next prompt

# Display results
for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
    print(f"\n{i}. {prompt}")
    print(f"   Answer: {response[:100]}...")
```

### **Audio Processing Pipeline**

```python
import regolo
import os

regolo.default_key = "<YOUR_API_KEY>"
regolo.default_audio_transcription_model = "faster-whisper-large-v3"
regolo.default_chat_model = "Llama-3.3-70B-Instruct"

# 1. Transcribe audio
audio_client = regolo.RegoloClient()
transcription = audio_client.audio_transcription(
    file="meeting_recording.mp3",
    language="en",
    response_format="json"
)

print("Transcription:", transcription)

# 2. Summarize transcription
chat_client = regolo.RegoloClient()
summary = chat_client.run_chat(
    user_prompt=f"Summarize this meeting transcript: {transcription}"
)

print("\nSummary:", summary[1])

# 3. Extract action items
action_items = chat_client.run_chat(
    user_prompt="List the action items from this meeting as bullet points"
)

print("\nAction Items:", action_items[1])
```

### **Model Management Automation**

```python
from regolo.cli import ModelManagementClient
import time

def deploy_model_workflow(model_name, hf_url, gpu_preference="ECS1GPU11"):
    """Complete workflow to deploy a HuggingFace model"""

    client = ModelManagementClient()

    # 1. Authenticate
    print("Authenticating...")
    client.authenticate("username", "password")

    # 2. Register model
    print(f"Registering model: {model_name}")
    try:
        client.register_model(
            name=model_name,
            is_huggingface=True,
            url=hf_url
        )
        print("✓ Model registered")
    except Exception as e:
        if "already exists" in str(e):
            print("⚠ Model already registered")
        else:
            raise

    # 3. Check GPU availability
    print("Checking GPU availability...")
    gpus = client.get_available_gpus()

    available_gpu = None
    for gpu in gpus['gpus']:
        if gpu['InstanceType'] == gpu_preference:
            available_gpu = gpu_preference
            break

    if not available_gpu and gpus['gpus']:
        available_gpu = gpus['gpus'][0]['InstanceType']

    if not available_gpu:
        raise Exception("No GPUs available")

    print(f"Using GPU: {available_gpu}")

    # 4. Load model for inference
    print("Loading model for inference...")
    vllm_config = {
        "max_model_len": 2048,
        "gpu_memory_utilization": 0.85,
        "tensor_parallel_size": 1
    }

    result = client.load_model_for_inference(
        model_name=model_name,
        gpu=available_gpu,
        vllm_config=vllm_config
    )

    if result.get('success'):
        print("✓ Model loading initiated")
    else:
        print("⚠ Model may already be loaded")

    # 5. Wait and verify
    print("Waiting for model to load (60s)...")
    time.sleep(60)

    loaded = client.get_loaded_models()
    model_loaded = any(
        m['model_name'] == model_name
        for m in loaded.get('loaded_models', [])
    )

    if model_loaded:
        print("✓ Model successfully loaded and ready for inference")
    else:
        print("⚠ Model not yet loaded, may need more time")

    # 6. Show status
    status = client.get_user_inference_status()
    print(f"\nCurrent status:")
    print(f"  Active models: {loaded.get('total', 0)}")
    print(f"  Month cost: €{status.get('total_cost', 0):.2f}")

    return client

# Usage
client = deploy_model_workflow(
    model_name="my-gpt2",
    hf_url="gpt2",
    gpu_preference="ECS1GPU11"
)
```

### **Cost Monitoring Script**

```python
from regolo.cli import ModelManagementClient
from datetime import datetime
import time

def monitor_costs_continuously(threshold_eur=100, check_interval=3600):
    """Monitor costs and alert when threshold exceeded"""

    client = ModelManagementClient()
    client.authenticate("username", "password")

    print(f"Monitoring costs. Alert threshold: €{threshold_eur}")
    print(f"Check interval: {check_interval}s")

    while True:
        try:
            # Get current month status
            status = client.get_user_inference_status()
            current_cost = status.get('total_cost', 0)

            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
            print(f"Current month cost: €{current_cost:.2f}")

            # Check loaded models
            loaded = client.get_loaded_models()
            if loaded['loaded_models']:
                print(f"Active models: {loaded['total']}")
                for model in loaded['loaded_models']:
                    print(f"  - {model['model_name']}: €{model['cost']:.2f}")

            # Alert if threshold exceeded
            if current_cost >= threshold_eur:
                print(f"\n⚠️  ALERT: Cost threshold exceeded!")
                print(f"Current: €{current_cost:.2f} / Threshold: €{threshold_eur}")

                # List recommendations
                if loaded['loaded_models']:
                    print("\nConsider unloading these models:")
                    for model in loaded['loaded_models']:
                        print(f"  - {model['model_name']} (Session: {model['session_id']})")

                break

            # Sleep until next check
            time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Error during monitoring: {e}")
            time.sleep(60)  # Wait a minute before retrying

# Run monitoring (checks every hour)
# monitor_costs_continuously(threshold_eur=100, check_interval=3600)
```

---

## **Best Practices**

### **API Key Security**

1. **Never hardcode API keys** in your source code
2. **Use environment variables** or secure key management
3. **Rotate keys regularly** for security

```python
import os
import regolo

# Load key from environment
regolo.default_key = os.getenv("REGOLO_API_KEY")

# Hardcoded key
regolo.default_key = "sk-xxxxxxxxxxxxx"
```

### **Error Handling**

```python
import regolo
from httpx import HTTPStatusError

regolo.default_key = "<YOUR_API_KEY>"

client = regolo.RegoloClient()

try:
    response = client.run_chat(user_prompt="Hello")
    print(response)
except HTTPStatusError as e:
    if e.response.status_code == 401:
        print("Authentication failed. Check your API key.")
    elif e.response.status_code == 429:
        print("Rate limit exceeded. Please wait before retrying.")
    else:
        print(f"HTTP error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### **Resource Management**

```python
import regolo

# Use context managers when working with files
client = regolo.RegoloClient()

# Audio transcription
with open("audio.mp3", "rb") as audio_file:
    transcription = client.audio_transcription(file=audio_file)

# Always clear conversations when starting new topics
client.run_chat(user_prompt="Tell me about Python")
client.clear_conversations()  # Clear before new topic
client.run_chat(user_prompt="Tell me about JavaScript")
```

### **Model Naming Conventions**

- Use descriptive, lowercase names: `my-bert-model`
- Include version numbers: `gpt2-v1`, `llama-2-7b`
- Avoid special characters except hyphens and underscores
- Keep names under 200 characters
- Don't use GitLab reserved names

---

## **Troubleshooting**

### **Common Issues**

#### **Authentication Errors**

```python
# Problem: "API key is required"
# Solution: Set the API key
import regolo
regolo.default_key = "<YOUR_API_KEY>"

# Or pass directly
client = regolo.RegoloClient(api_key="<YOUR_API_KEY>")
```

#### **Model Not Found**

```bash
# Problem: Model not found when loading for inference
# Solution: Register the model first
regolo models register --name my-model --type huggingface --url gpt2
```

#### **SSH Authentication Failed**

```bash
# Problem: Cannot push to GitLab repository
# Solution: Add your SSH key
regolo ssh add --title "My Key" --key-file ~/.ssh/id_rsa.pub

# Test SSH connection
ssh -T git@gitlab.regolo.ai
```

#### **Model Loading Timeout**

```bash
# Problem: Model takes too long to load
# Solution: Large models need time, check status periodically
regolo inference status

# Wait and check again
sleep 60
regolo inference status
```

#### **High Costs**

```bash
# Problem: Unexpected high costs
# Solution: Check loaded models and unload unused ones
regolo inference status
regolo inference unload --model-name unused-model

# Check detailed cost breakdown
regolo inference user-status --month 012025
```

### **Getting Help**

For additional support:

1. Check the [API documentation](https://devmid.regolo.ai/docs)
2. View CLI help: `regolo --help` or `regolo <command> --help`
3. Contact Regolo support through your organization

---

## **API Reference**

### **RegoloClient Methods**

| Method                                           | Description                   |
|--------------------------------------------------|-------------------------------|
| `completions(prompt, stream, max_tokens, ...)`   | Generate text completion      |
| `run_chat(user_prompt, stream, max_tokens, ...)` | Run chat completion           |
| `add_prompt_to_chat(prompt, role)`               | Add message to conversation   |
| `clear_conversations()`                          | Clear conversation history    |
| `create_image(prompt, n, quality, size, ...)`    | Generate images               |
| `audio_transcription(file, language, ...)`       | Transcribe audio              |
| `embeddings(input_text)`                         | Generate text embeddings      |
| `rerank(query, documents, top_n, ...)`           | Rerank documents by relevance |
| `change_model(model)`                            | Switch to different model     |

### **Static Methods**

| Method                                            | Description              |
|---------------------------------------------------|--------------------------|
| `static_completions(prompt, model, api_key, ...)` | Static completion method |
| `static_chat_completions(messages, model, ...)`   | Static chat method       |
| `static_image_create(prompt, model, ...)`         | Static image generation  |
| `static_audio_transcription(file, model, ...)`    | Static transcription     |
| `static_embeddings(input_text, model, ...)`       | Static embeddings        |
| `get_available_models(api_key, model_info)`       | List available models    |

### **CLI Commands Reference**

| Command                         | Description                   |
|---------------------------------|-------------------------------|
| `regolo auth login`             | Authenticate with credentials |
| `regolo auth logout`            | Clear authentication tokens   |
| `regolo models register`        | Register a new model          |
| `regolo models list`            | List registered models        |
| `regolo models details <name>`  | Get model details             |
| `regolo models delete <name>`   | Delete a model                |
| `regolo ssh add`                | Add SSH key                   |
| `regolo ssh list`               | List SSH keys                 |
| `regolo ssh delete <id>`        | Delete SSH key                |
| `regolo inference gpus`         | List available GPUs           |
| `regolo inference load <model>` | Load model for inference      |
| `regolo inference unload`       | Unload model                  |
| `regolo inference status`       | Show loaded models            |
| `regolo inference user-status`  | Show cost/billing info        |
| `regolo chat`                   | Interactive chat              |
| `regolo get-available-models`   | List API models               |
| `regolo create-image`           | Generate images               |
| `regolo transcribe-audio`       | Transcribe audio              |
| `regolo rerank`                 | Rerank documents              |

---

## **Version Information**

- **Client Version**: Check with `pip show regolo`
- **API Version**: 1.0.0
- **Python Requirement**: >= 3.8
- **Management API Base URL**: `https://devmid.regolo.ai`
- **Inference API Base URL**: `https://api.regolo.ai`

---

For more information,
visit the [Regolo.ai documentation](https://docs.regolo.ai) or contact support through your organization.
