import typer
from epidemiqs import __version__

app = typer.Typer(help="Version")

@app.command()
def show():
    print(f"EpidemIQs version {__version__}")
