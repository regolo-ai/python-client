A simple python client for regolo.ai




# BASIC USAGE

### Import the modules

    from httpx import stream
    
    from regolo.client.regolo_client import RegoloClient
    from regolo.instance.structures.conversation_model import Conversation, ConversationLine
    import regolo



### Add default key and model to use by default when creating instances of RegoloClient or when running static_completions and static_chat_completions

    regolo.default_key = "<EXAMPLE_KEY>"
    regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"

### set full output parameter

    full_output = False

## Use the module

### STREAMING USAGE


#### Completions with stream

    client = RegoloClient()
    res = client.completions("Tell me about Rome in a concise manner", stream=True, full_output=full_output, max_tokens=200)
    
    if not full_output:
        while True:
            try:
                print(next(res), end='', flush=True)
            except StopIteration:
                break
    else:
        while True:
            try:
                print(next(res))
            except StopIteration:
                break

#### chat completions with stream


    client = RegoloClient(api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct")
    response = client.run_chat(user_prompt="Tell me about Rome in a concise manner", max_tokens=200, stream=True, full_output=full_output)
    
    if not full_output:
        while True:
            try:
                res = next(response)
                if res[0]:
                    print(res[0] + ":")
                print(res[1], end="", flush=True)
            except StopIteration:
                break
    else:
        while True:
            try:
                res = next(response)
                print(res)
            except StopIteration:
                break


### NON-STREAMING USAGE



#### example completions
    
    client = RegoloClient()
    print(client.static_completions('tell me about Rome', max_tokens=200, api_key="<EXAMPLE_KEY>",
                                model="meta-llama/Llama-3.3-70B-Instruct", full_output=full_output))
    
    
    
    client = RegoloClient(api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct")
    print(client.completions("Tell me about Rome in a concise manner", full_output=full_output))




#### example chat completions
    
#### query endpoint directly:
    client = RegoloClient(api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct")
    print(client.static_chat_completions([{"role": "user", "content": "How are you?"}], api_key="<EXAMPLE_KEY>",
                                 model="meta-llama/Llama-3.3-70B-Instruct", full_output=full_output))
#### query with regolo_client memory
    "EQUIVALENT APRROACHES:"
    1.        
        # Chat adding full prompts with add_prompt_to_chat():
    
        client.add_prompt_to_chat(prompt="Tell me about Rome in a concise manner", role="user")  # allows to add instruction as any role
        response = client.run_chat(full_output=full_output)
        if full_output:
            print(response)
        else:
            print(f"{response[0]}: {response[1]}")
        
        # print(client.instance.get_conversation())
        
        client.add_prompt_to_chat(prompt="tell a little bit more!", role="user")
        response = client.run_chat(full_output=full_output)
        if full_output:
            print(response)
        else:
            print(f"{response[0]}: {response[1]}")
        
        # print(client.instance.get_conversation())
    
    2.
        # Chat handling user prompt directly in run_chat()
        
        client = RegoloClient(api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct")
        response = client.run_chat(user_prompt="Tell me about Rome in a concise manner", full_output=full_output)
        
        if full_output:
            print(response)
        else:
            print(f"{response[0]}: {response[1]}")
        
        response = client.run_chat(user_prompt="Tell me a little bit more in a concise manner", full_output=full_output)
        
        if full_output:
            print(response)
        else:
            print(f"{response[0]}: {response[1]}")