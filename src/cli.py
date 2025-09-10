import typer
from rich import print
from orchestrator import AIProvider, orchestrate_chat

app = typer.Typer()

@app.command()
def chat(
    prompt: str,
    provider: AIProvider = typer.Option(
        AIProvider.OPENAI,
        "--provider",
        "-p",
        help="The AI provider to use.",
        case_sensitive=False,
    ),
):
    """
    Have a conversation with an AI model from a specified provider.
    """
    print(f"[bold green]You:[/bold green] {prompt}")

    response = orchestrate_chat(prompt, provider)

    if response:
        print(f"[bold blue]{provider.value.title()}:[/bold blue] {response}")
    else:
        print(f"[bold red]Failed to get a response from {provider.value}.[/bold red]")

if __name__ == "__main__":
    app()
