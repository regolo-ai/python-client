from pygments.lexer import default

import regolo
import click
import pprint

from regolo import RegoloClient

@click.group()
def cli():
    pass

@click.command()
@click.option('--no-hide', is_flag=True, default=False, help='Do not hide the API key when typing')
@click.option('--disable-newlines', is_flag=True, default=False, help='Disable new lines, they will be replaced with space character')
def chat(no_hide, disable_newlines):


    API_KEY = click.prompt("Insert your regolo API key", hide_input=not no_hide)
    available_models = regolo.RegoloClient.get_available_models(API_KEY)

    if len(available_models) == 0:
        click.echo("No models available with your API key.")
        exit(1)

    available_models_dict = {}
    model_number = 1
    for model in available_models:
        available_models_dict[model_number] = model

    click.echo(f"The models you can use are:\n {pprint.pformat(available_models_dict)}")

    prompt_model_number = click.prompt("Write the number of the model to use")

    try:
        model = available_models_dict[int(prompt_model_number)]
    except ValueError:
        raise Exception("Not a number")
    except KeyError:
        raise Exception("Not a valid number.")

    click.echo(f"\n")

    client = RegoloClient(api_key=API_KEY, model=model)
    click.echo(f"You can now chat with {model}, write \"bye\" to exit")

    while True:
        # ask user input
        user_input = click.prompt("user")

        if user_input == "bye":
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
cli.add_command(chat)