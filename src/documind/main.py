"""CLI entrypoint for DocuMind."""

import uvicorn

from documind.app import create_app
from documind.config import Settings


def cli() -> None:
    cfg = Settings()
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    cli()
