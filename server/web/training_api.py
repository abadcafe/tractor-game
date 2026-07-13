"""HTTP adapter for standalone training initialization and control."""

from __future__ import annotations

import asyncio
from functools import partial
from pathlib import Path
from typing import Annotated, Never

from anyio import to_thread
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, BeforeValidator, ConfigDict

from server.foundation.result import Rejected
from server.training_control.cli_client import TrainingProcess
from server.training_control.commands import (
    TrainingInitRequest,
    TrainingResumeRequest,
    init_command_argv,
    resume_command_argv,
)
from server.training_control.init_control import TrainingInitialization
from server.training_control.metric_queries import (
    TrainingMetrics,
    query_training_metrics,
)
from server.training_control.process_control import StopResult
from server.web.state import ServerState


def _parse_request_path(value: object) -> object:
    if isinstance(value, str):
        return Path(value)
    return value


type _RequestPath = Annotated[
    Path, BeforeValidator(_parse_request_path)
]


class TrainingRunBody(BaseModel):
    """Optional run directory used by control commands."""

    model_config = ConfigDict(extra="forbid", strict=True)

    run_dir: _RequestPath | None = None


class TrainingControlConfigResponse(BaseModel):
    """Server defaults consumed by the training SPA."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    default_run_dir: Path
    stop_timeout_seconds: float


def register_training_routes(app: FastAPI, state: ServerState) -> None:
    async def training_config() -> TrainingControlConfigResponse:
        config = state.training_control_config
        return TrainingControlConfigResponse(
            default_run_dir=config.default_run_dir,
            stop_timeout_seconds=config.stop_timeout_seconds,
        )

    async def init_schema() -> object:
        return TrainingInitRequest.model_json_schema()

    async def resume_schema() -> object:
        return TrainingResumeRequest.model_json_schema()

    async def initialize_training(
        request: TrainingInitRequest,
    ) -> TrainingInitialization:
        run_dir = state.training_control_config.resolve_run_dir(
            request.run_dir
        )
        summary_result = await state.training_cli_client.summary(
            run_dir
        )
        if isinstance(summary_result, Rejected):
            _raise_rejected(summary_result, status_code=409)
        run_status = summary_result.value
        if run_status.process is not None:
            raise HTTPException(
                status_code=409,
                detail="training process is already running",
            )
        if (
            run_status.state in ("READY", "BROKEN")
            and request.replace_existing != "yes"
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "type yes to replace existing training artifacts"
                ),
            )
        resolved_request = request.model_copy(
            update={"run_dir": run_dir}
        )
        result = await asyncio.shield(
            state.training_init_control.initialize(
                run_dir=run_dir,
                command=init_command_argv(resolved_request),
                working_directory=Path.cwd(),
            )
        )
        if isinstance(result, Rejected):
            _raise_rejected(result, status_code=409)
        return result.value

    async def resume_training(
        request: TrainingResumeRequest,
    ) -> TrainingProcess:
        run_dir = state.training_control_config.resolve_run_dir(
            request.run_dir
        )
        summary_result = await state.training_cli_client.summary(
            run_dir
        )
        if isinstance(summary_result, Rejected):
            _raise_rejected(summary_result, status_code=409)
        run_status = summary_result.value
        if run_status.state != "READY":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"training run is not ready: {run_status.state}"
                ),
            )
        resolved_request = request.model_copy(
            update={"run_dir": run_dir}
        )
        result = await state.training_process_control.start(
            run_dir=run_dir,
            command=resume_command_argv(resolved_request),
            working_directory=Path.cwd(),
        )
        if isinstance(result, Rejected):
            _raise_rejected(result, status_code=409)
        return result.value

    async def stop_training(body: TrainingRunBody) -> StopResult:
        run_dir = state.training_control_config.resolve_run_dir(
            body.run_dir
        )
        result = await asyncio.shield(
            state.training_process_control.stop(
                run_dir=run_dir,
                timeout_seconds=(
                    state.training_control_config.stop_timeout_seconds
                ),
            )
        )
        if isinstance(result, Rejected):
            _raise_rejected(result, status_code=409)
        return result.value

    async def training_summary(
        run_dir: Path | None = None,
    ) -> object:
        result = await state.training_cli_client.summary(
            state.training_control_config.resolve_run_dir(run_dir)
        )
        if isinstance(result, Rejected):
            _raise_rejected(result, status_code=409)
        return result.value

    async def training_metrics(
        run_dir: Path | None = None,
        session_id: str | None = None,
        update_limit: Annotated[int, Query(ge=1, le=5000)] = 500,
        series_points: Annotated[int, Query(ge=1, le=1000)] = 500,
    ) -> TrainingMetrics:
        result = await to_thread.run_sync(
            partial(
                query_training_metrics,
                state.training_control_config.resolve_run_dir(run_dir),
                session_id=session_id,
                update_limit=update_limit,
                series_points=series_points,
            )
        )
        if isinstance(result, Rejected):
            _raise_rejected(result, status_code=409)
        return result.value

    app.add_api_route(
        "/api/training/config", training_config, methods=["GET"]
    )
    app.add_api_route(
        "/api/training/init/schema", init_schema, methods=["GET"]
    )
    app.add_api_route(
        "/api/training/resume/schema", resume_schema, methods=["GET"]
    )
    app.add_api_route(
        "/api/training/init", initialize_training, methods=["POST"]
    )
    app.add_api_route(
        "/api/training/resume", resume_training, methods=["POST"]
    )
    app.add_api_route(
        "/api/training/stop", stop_training, methods=["POST"]
    )
    app.add_api_route(
        "/api/training/summary", training_summary, methods=["GET"]
    )
    app.add_api_route(
        "/api/training/metrics", training_metrics, methods=["GET"]
    )


def _raise_rejected(result: Rejected, *, status_code: int) -> Never:
    raise HTTPException(status_code=status_code, detail=result.reason)
