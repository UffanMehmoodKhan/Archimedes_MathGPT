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

@app.command()
def infer(question: str):
    """Generate an answer for a single math question."""
    typer.echo(f"Question: {question}")
    answer = answer_question(question)
    typer.echo(f"Predicted Answer: {answer}")

if __name__ == "__main__":
    app()
