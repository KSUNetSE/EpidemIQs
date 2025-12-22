import typer
from . import run, version

app = typer.Typer(help="EpidemIQs CLI – run epidemic modeling workflows")
app.add_typer(run.app, name="run")
app.add_typer(version.app, name="version")

# python -m epidemiqs.cli works too:
def main():
    app()