import json
import os
import pprint
from datetime import datetime
from io import BytesIO
from typing import Any

import click
from PIL import Image

import regolo
from regolo import RegoloClient
from regolo.keys.keys import KeysHandler

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
AUDIO_EXTENSIONS = ["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"]


@click.group()
def cli():
    pass

@click.command("get-available-models", help="Gets available models")
@click.option('--api-key', required=True, help='The API key used to query Regolo.')
@click.option('--model-type', default="", required=False, type=click.Choice(['', 'chat', 'image_generation', "embedding", "audio_transcription"]), help='Thee type of the models you want to retrieve (returns all by default)')
def get_available_models(api_key: str, model_type: str):
    available_models: list[dict] = regolo.RegoloClient.get_available_models(api_key, model_info=True)
    output_models: list[tuple] = []
    for model in available_models:
        if model_type in model["model_info"]["mode"]:
            output_models.append((model["model_name"], model["model_info"]["mode"]))
    click.echo(pprint.pformat(output_models))


@click.command("chat", help="Allows chatting with LLMs")
@click.option('--no-hide',required=False, is_flag=True, default=False, help='Do not hide the API key when typing')
@click.option('--api-key', required=False, help='The API key used to chat with Regolo.')
@click.option('--disable-newlines', required=False, is_flag=True, default=False,
              help='Disable new lines, they will be replaced with space character')
def chat(no_hide: bool, api_key:str, disable_newlines: bool):
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
@click.option('--prompt', default="A generic image", help='The text prompt for image generation. (Defaults to "A generic image")')
@click.option('--n', default=1, help='The number of images to generate. (Defaults to 1)')
@click.option('--quality', default="standard", help="The quality of the image that will be generated. The 'hd' value creates images with finer details and greater consistency across the image. (Defaults to 'standard'')")
@click.option('--size', default="1024x1024", help="The size of the generated images. (Defaults to '1024x1024')")
@click.option('--style', default="realistic", help="The style of the generated images. (Defaults to 'realistic')")

def create_image(api_key:str, save_path: str, model: str, prompt: str, n: int, quality: str, size: str, style: str):

    if model is None:
        raise Exception("You must specify a model")

    KeysHandler.check_key(api_key)

    if save_path is None:
        save_path = os.path.join(os.getcwd(), "images")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    client = RegoloClient(api_key=api_key, image_generation_model=model)

    images_bytes = client.create_image(prompt=prompt, n=n, quality=quality, size=size, style=style)

    for image_bytes in images_bytes:
        image = Image.open(BytesIO(image_bytes))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(save_path, f"{timestamp}.png")

        # If it already exists, use _1, _2, etc.
        i = 1
        while os.path.exists(filepath):
            filename = f"{timestamp}_{i}{IMAGE_EXTENSIONS[0]}"
            filepath = os.path.join(save_path, filename)
            i += 1

        image.save(filepath)


@click.command("transcribe-audio", help='Transcribes audio files')
@click.option('--api-key', required=True, help='The API key used to transcribe.')
@click.option('--model', required=True, help='The model to use for transcription (gpt-4o-transcribe, gpt-4o-mini-transcribe, or whisper-1)')
@click.option('--file-path', required=True, help='Path to the audio file to transcribe')
@click.option('--save-path', help='Path to save the transcription (prints to console if not specified)')
@click.option('--language', help='Language of the input audio in ISO-639-1 format (e.g., en, es, fr)')
@click.option('--prompt', help='Optional text to guide the model\'s style or continue a previous audio segment')
@click.option('--response-format', default="json", type=click.Choice(['json', 'text', 'srt', 'verbose_json', 'vtt']), help='Format of the transcript output (defaults to json)')
@click.option('--temperature', type=float, help='Sampling temperature between 0 and 1 (defaults to 0)')
@click.option('--chunking-strategy', help='Controls audio chunking: "auto" or JSON object string')
@click.option('--include-logprobs', is_flag=True, help='Include log probabilities (only works with json format and gpt-4o models)')
@click.option('--timestamp-granularities', multiple=True, type=click.Choice(['word', 'segment']), help='Timestamp granularities (requires verbose_json format)')
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


cli.add_command(transcribe_audio)
cli.add_command(chat)
cli.add_command(create_image)
cli.add_command(get_available_models)
