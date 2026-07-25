"""Run the Tractor game web server."""

from __future__ import annotations

import argparse
import socket
from collections.abc import Sequence

import uvicorn
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from server.web.app import WebApplication
from server.web.logging_config import configure_server_logging


class _Options(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)


class _Server(uvicorn.Server):
    def __init__(
        self,
        config: uvicorn.Config,
        application: WebApplication,
    ) -> None:
        super().__init__(config)
        self._application = application

    async def shutdown(
        self, sockets: list[socket.socket] | None = None
    ) -> None:
        self._application.begin_shutdown()
        await super().shutdown(sockets)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m server.web")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    return parser


def _main(arguments: Sequence[str] | None = None) -> None:
    parser = _argument_parser()
    try:
        options = _Options.model_validate(
            vars(parser.parse_args(arguments))
        )
    except ValidationError as error:
        parser.error(error.errors(include_url=False)[0]["msg"])
    configure_server_logging()
    application = WebApplication()
    config = uvicorn.Config(
        application.asgi,
        host=options.host,
        port=options.port,
        ws="websockets-sansio",
        lifespan="on",
    )
    server = _Server(config, application)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    if not server.started:
        raise SystemExit(1)


_main()
