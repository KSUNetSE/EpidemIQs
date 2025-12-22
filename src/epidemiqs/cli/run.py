import typer
from epidemiqs.main import Epidemiqs
from epidemiqs.utils.config import get_settings,Settings
import asyncio
app = typer.Typer(help="EpidemIQs CLI – run epidemic modeling workflows")

@app.command()
def now(config: str = typer.Option("config.yaml", help="Path to your YAML config")):
    """Run the EpidemIQs workflow."""
    cfg = get_settings(config_path=config)
    epidemiqs_instance = Epidemiqs(cfg=cfg)
    print("Starting the EpidemIQs workflow...")
    asyncio.run( epidemiqs_instance.run() )
    print("EpidemIQs workflow completed.")

if __name__ == "__main__":
    app()