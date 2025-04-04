import os

import regolo
import click
import pprint
from PIL import Image
from io import BytesIO
from datetime import datetime

from regolo import RegoloClient

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]


@click.group()
def cli():
    pass

@click.command("get-available-models")
@click.option('--api-key', is_flag=True, default=False, help='The API key used to query Regolo.')
def get_available_models(api_key):
    available_models: list[dict] = regolo.RegoloClient.get_available_models(api_key, model_info=True)
    output_models: list[tuple] = []
    for model in available_models:
        output_models.append((model["model_name"], model["model_info"]["mode"]))
    click.echo(output_models)


@click.command()
@click.option('--no-hide', is_flag=True, default=False, help='Do not hide the API key when typing')
@click.option('--disable-newlines', is_flag=True, default=False,
              help='Disable new lines, they will be replaced with space character')
def chat(no_hide, disable_newlines):
    api_key = click.prompt("Insert your regolo API key", hide_input=not no_hide)
    available_models: list[dict] = regolo.RegoloClient.get_available_models(api_key, model_info=True)

    if len(available_models) == 0:
        click.echo("No models available with your API key.")
        exit(1)

    available_models_dict = {}
    model_number = 1
    for model in available_models:
        if model["model_info"]["mode"] is None:
            available_models_dict[model_number] = model
            available_models_dict[model_number] = model["model_name"]
            model_number += 1

    click.echo(f"The models you can use are:\n {pprint.pformat(available_models_dict)}")

    prompt_model_number = click.prompt("Write the number of the model to use")

    try:
        model = available_models_dict[int(prompt_model_number)]
    except ValueError:
        raise Exception("Not a number")
    except KeyError:
        raise Exception("Not a valid number.")

    click.echo(f"\n")

    client = RegoloClient(api_key=api_key, model=model)
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


@click.command("create-image")
@click.option('--api-key', is_flag=True, default=False, help='The API key used generate.')
@click.option('--save-path', default=False, help='The path in which to save the images')
@click.option('--model', default="realistic", help="The text prompt for image generation.")
@click.option('--prompt', default=1, help='The number of images to generate. (Defaults to 1)')
@click.option('--n', default=1, help='The number of images to generate. (Defaults to 1)')
@click.option('--quality', default="standard", help="The quality of the image that will be generated. The 'hd' value creates images with finer details and greater consistency across the image. (Defaults to 'standard'')")
@click.option('--size', default="1024x1024", help="The size of the generated images. (Defaults to '1024x1024')")
@click.option('--style', default="realistic", help="The style of the generated images. (Defaults to 'realistic')")

def create_image(api_key:str, save_path: str, model: str, prompt: str, n: int, quality: str, size: str, style: str):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    client = RegoloClient(api_key=api_key, model=model)

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



cli.add_command(chat)
