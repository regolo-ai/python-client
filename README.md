# **Regolo.ai Python Client**

A simple Python client for interacting for **Regolo.ai's** LLM-based API.

## **Installation**
Ensure you have the `regolo` module installed. If not, install it using:

```bash
  pip install regolo
```

# **Basic Usage**

## **1. Import the regolo module**

```python
import regolo
```
 ## **2. Set Up Default API Key and Model**

To avoid manually passing the API key and model in every request, you can set them globally:

```python
regolo.default_key = "<EXAMPLE_KEY>"
regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"
```

This ensures that all `RegoloClient` instances and static functions will
use the specified API key and model.

Still, you can create run methods by inserting model and key directly.

 ## **3. Perform a basic request**

### Completion:
```python
print(regolo.static_completions(prompt="Tell me something about Rome."))
```

### Chat_completion
```python
print(regolo.static_chat_completions(messages=[{"role": "user", "content": "Tell me something about rome"}]))
```

---
# **Loading envs**

#### if you want to interact with this client through environment variables, you can follow this reference:

### Default values

- "API_KEY"

You can use this environment variable to insert the default_key.
You can load it after importing regolo using regolo.key_load_from_env_if_exists().
Using it is equivalent to updating regolo.default_key when you import regolo.

- "LLM"

You can use this environment variable to insert the default_model.
You can load it after importing regolo using regolo.default_model_load_from_env_if_exists().
This is equivalent to updating regolo.default_model when you import regolo.

- "IMAGE_MODEL"

You can use this environment variable to insert the default_image_model.
You can load it after importing regolo using regolo.default_image_load_from_env_if_exists().
This is equivalent to updating regolo.default_image_model when you import regolo.

- "EMBEDDER_MODEL"

You can use this environment variable to insert the default_embedder_model.
You can load it after importing regolo using regolo.default_embedder_load_from_env_if_exists().
This is equivalent to updating regolo.default_embedder_model when you import regolo.


> [!TIP]
> All the environment variables that are equivalent to a default value in regolo __init__ can be updated together
> through regolo.try_loading_from_env().
> 
> It does nothing but run all the load_from_env methods al once.

### Endpoints

- "REGOLO_URL"

You can use this env variable to set the default base_url used by regolo client and its static methods.

- "COMPLETIONS_URL_PATH"

You can use this env variable to set the base_url used by regolo client and its static methods.

- "CHAT_COMPLETIONS_URL_PATH"

You can use this env variable to set the chat completions endpoint used by regolo client and its static methods.

- "IMAGE_GENERATION_URL_PATH"

You can use this env variable to set the image generation endpoint used by regolo client and its static methods.

- "EMBEDDINGS_URL_PATH"

You can use this env variable to set the embedding generation endpoint used by regolo client and its static methods.


> [!TIP]
> The endpoints environment variables can be changed during execution.
> Since the client works directly with them.
> 
> However, you are likely not to want to change them, since they are tied to how we handle our endpoints. 

---

# **Other usages**

## **Handling streams**


**With full output:**

```python
import regolo
regolo.default_key = "<EXAMPLE_KEY>"
regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"

# Completions

client = regolo.RegoloClient()
response = client.completions("Tell me about Rome in a concise manner", full_output=True, stream=True)

while True:
    try:
        print(next(response))
    except StopIteration:
        break

# Chat completions

client = regolo.RegoloClient()
response = client.run_chat(user_prompt="Tell me about Rome in a concise manner", full_output=True, stream=True)


while True:
    try:
        print(next(response))
    except StopIteration:
        break
```

**Without full output:**

```python
import regolo
regolo.default_key = "<EXAMPLE_KEY>"
regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"

# Completions

client = regolo.RegoloClient()
response = client.completions("Tell me about Rome in a concise manner", full_output=True, stream=True)

while True:
    try:
        print(next(response), end='', flush=True)
    except StopIteration:
        break

# Chat completions

client = regolo.RegoloClient()
response = client.run_chat(user_prompt="Tell me about Rome in a concise manner", full_output=True, stream=True)

while True:
    try:
        res = next(response)
        if res[0]:
            print(res[0] + ":")
        print(res[1], end="", flush=True)
    except StopIteration:
        break
```

## **Handling chat through add_prompt_to_chat()**

```python
import regolo

regolo.default_key = "<EXAMPLE_KEY>"
regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"

client = regolo.RegoloClient()

# Make a request

client.add_prompt_to_chat(role="user", prompt="Tell me about rome!")

print(client.run_chat())

# Continue the conversation

client.add_prompt_to_chat(role="user", prompt="Tell me something more about it!")

print(client.run_chat())

# You can print the whole conversation if needed

print(client.instance.get_conversation())
```

It is to consider that using the user_prompt parameter in run_chat() is equivalent to adding a prompt with role=user
through add_prompt_to_chat().


## **Handling image models**

**Without client:**
```python
from io import BytesIO

import regolo
from PIL import Image

regolo.default_image_model = "FLUX.1-dev"
regolo.default_key = "<EXAMPLE_KEY>"

img_bytes = regolo.static_image_create(prompt="a cat")[0]

image = Image.open(BytesIO(img_bytes))

image.show()
```

**With client**
```python
from io import BytesIO

import regolo
from PIL import Image
client = regolo.RegoloClient(image_model="FLUX.1-dev", api_key="<EXAMPLE_KEY>")

img_bytes = client.create_image(prompt="A cat in Rome")[0]

image = Image.open(BytesIO(img_bytes))

image.show()
```

## **Handling embedder models**

**Without client:**
```python
import regolo

regolo.default_key = "<EXAMPLE_KEY>"
regolo.default_embedder_model = "gte-Qwen2"


embeddings = regolo.static_embeddings(input_text=["test", "test1"])

print(embeddings)
```

**With client:**
```python
import regolo

client = regolo.RegoloClient(api_key="<EXAMPLE_KEY>", embedder_model="gte-Qwen2")

embeddings = client.embeddings(input_text=["test", "test1"])

print(embeddings)
```
