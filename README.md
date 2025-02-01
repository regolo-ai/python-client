A simple python client for regolo.ai




# BASIC USAGE

### Import the modules

    from httpx import stream
    
    from regolo.client.regolo_client import RegoloClient
    from regolo.instance.structures.conversation_model import Conversation, ConversationLine
    import regolo



### Add default key and model to use by default when creating instances of RegoloClient or running static_completions and static_chat_completions

    regolo.default_key = "<EXAMPLE_KEY>"
    regolo.default_model = "meta-llama/Llama-3.3-70B-Instruct"


## Use the module

### STREAMING USAGE


#### Completions with stream

     client = RegoloClient()
     res = client.completions("Tell me about Rome in a concise manner", stream=True, full_output=False, max_tokens=200)
    
     while True:
         try:
             print(next(res), end='', flush=True)
         except StopIteration:
             break

#### chat completions with stream


     client = RegoloClient(api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct")
     response = client.run_chat(user_prompt="Tell me about Rome in a concise manner", max_tokens=200, stream=True)
    
     while True:
         try:
             res = next(response)
             if res[0]:
                 print(res[0] + ":")
             print(res[1], end="", flush=True)
         except StopIteration:
             break


### NON-STREAMING USAGE



#### example completions
    
    query endpoint directly
        client = RegoloClient()
        print(client.static_completions('tell me about Rome', max_tokens=200, api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct")["choices"][0]["text"])

    query with regolo_client memory
        client = RegoloClient(api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct")
        print(client.completions("Tell me about Rome in a concise manner"))





#### example chat completions
    
#### query endpoint directly:
    client = RegoloClient(api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct")
    print(client.static_chat_completions([{"role": "user", "content": "How are you?"}], api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct"))

#### query with regolo_client memory
    "EQUIVALENTS:"
        
    # Chat adding full prompts:

        client = RegoloClient(api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct") 
        client.add_prompt_to_chat(prompt="Tell me about Rome in a concise manner", role="user") # allows to add instruction as any role
        print(client.run_chat())
        print(client.instance.get_conversation())
        client.add_prompt_to_chat(prompt="tell a little bit more!", role="user")
        print(client.run_chat())
        print(client.instance.get_conversation())
        
    # Chat handling user prompt directly in run_chat()

        client = RegoloClient(api_key="<EXAMPLE_KEY>", model="meta-llama/Llama-3.3-70B-Instruct") 
        print(client.run_chat(user_prompt="Tell me about Rome in a concise manner")) # can add user prompt directly in run_chat()
        print(client.run_chat(user_prompt="Tell me a little bit more in a concise manner"))
        