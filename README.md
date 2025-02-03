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

Still, you can create run methods by passing model and key directly.

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

client = regolo.RegoloClient()

# Make a request

client.add_prompt_to_chat(role="user", prompt="Tell me about rome!")

print(client.run_chat())

# Continue the conversation

client.add_prompt_to_chat(role="user", prompt="Tell me something more about it!")

print(client.run_chat())

# You can print the whole conversation if needed

print(print(client.instance.get_conversation()))
```

It is to consider that using the user_prompt parameter in run_chat() is equivalent to adding a prompt with role=user
through add_prompt_to_chat().