import typer
from app.train import train_model
from app.evaluate import evaluate_model
from app.inference import answer_question

app = typer.Typer()

@app.command()
def train(epochs: int = 3, batch_size: int = 4):
    """Train the Math Explanation Generator model."""
    typer.echo(f"Training model for {epochs} epochs with batch size {batch_size}...")
    train_model(epochs=epochs, batch_size=batch_size)
    typer.echo("Training finished.")

@app.command()
def evaluate():
    """Evaluate the trained model on the test set."""
    typer.echo("Evaluating the trained model...")
    evaluate_model()
    typer.echo("Evaluation completed.")

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()

@app.command()
def infer():
    """
    🌟 Interactive Math Explanation Chat Mode
    Type your questions and receive step-by-step explanations.
    Type 'exit' to quit.
    """
    console.rule("[bold cyan]🤖 Math Explanation Generator[/bold cyan]")
    console.print("[green]Ask me any math question![/green]")
    console.print("[dim]Type 'exit' or 'quit' to leave the session.[/dim]\n")

    while True:
        question = Prompt.ask("[bold yellow]❓ Enter your question[/bold yellow]")

        # exit condition
        if question.strip().lower() in ["exit", "quit"]:
            console.print("\n[bold red]👋 Exiting... Have a great day![/bold red]")
            break

        # reasoning spinner
        with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
            answer = answer_question(question)

        # pretty output
        console.print(
            Panel.fit(
                f"[bold cyan]Q:[/bold cyan] {question}",
                border_style="cyan",
                title="[bold]Your Question[/bold]"
            )
        )
        console.print(
            Panel.fit(
                f"[bold green]A:[/bold green] {answer}",
                border_style="magenta",
                title="[bold]Explanation[/bold]"
            )
        )
        console.print("\n")  # spacing


if __name__ == "__main__":
    app()
