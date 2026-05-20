from __future__ import annotations

import argparse
import logging

import uvicorn

from .config import get_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run the Irodori-API server.")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("api.app:app", host=str(args.host), port=int(args.port), reload=bool(args.reload))


if __name__ == "__main__":
    main()
