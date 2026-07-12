"""WebSocket stream for live training dashboard observations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict

from server.foundation import result as _result
from server.training_control.cli_client import TrainingCliSummary
from server.training_control.logs import read_log_tail
from server.web.state import ServerState

type _LogStream = Literal["stdout", "stderr"]

_PUSH_INTERVAL_SECONDS = 2.0
_LOG_MAX_BYTES = 200_000


class _TrainingStreamSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["snapshot"] = "snapshot"
    summary: TrainingCliSummary
    log_stream: _LogStream | None
    log_content: str | None


class _TrainingStreamError(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    type: Literal["error"] = "error"
    message: str


def register_training_stream_route(
    app: FastAPI,
    state: ServerState,
) -> None:
    """Register the single server-push dashboard observation route."""

    async def training_stream(
        websocket: WebSocket,
        run_dir: Path | None = None,
        metric_sequence: Annotated[int | None, Query(ge=0)] = None,
        telemetry_sequence: Annotated[int | None, Query(ge=0)] = None,
        log_stream: _LogStream | None = None,
    ) -> None:
        await _stream_training(
            websocket,
            state=state,
            run_dir=state.training_control_config.resolve_run_dir(
                run_dir
            ),
            metric_sequence=metric_sequence,
            telemetry_sequence=telemetry_sequence,
            log_stream=log_stream,
        )

    app.add_api_websocket_route("/ws/training", training_stream)


async def _stream_training(
    websocket: WebSocket,
    *,
    state: ServerState,
    run_dir: Path,
    metric_sequence: int | None,
    telemetry_sequence: int | None,
    log_stream: _LogStream | None,
) -> None:
    await websocket.accept()
    metric_cursor = metric_sequence
    telemetry_cursor = telemetry_sequence
    try:
        while True:
            snapshot_result = await _read_snapshot(
                state,
                run_dir=run_dir,
                metric_sequence=metric_cursor,
                telemetry_sequence=telemetry_cursor,
                log_stream=log_stream,
            )
            if isinstance(snapshot_result, _result.Rejected):
                message = _TrainingStreamError(
                    message=snapshot_result.reason
                )
            else:
                snapshot = snapshot_result.value
                message = snapshot
                if snapshot.summary.metrics:
                    metric_cursor = snapshot.summary.metrics[
                        -1
                    ].sequence
                if snapshot.summary.telemetry:
                    telemetry_cursor = snapshot.summary.telemetry[
                        -1
                    ].sequence
            await websocket.send_text(message.model_dump_json())
            await asyncio.sleep(_PUSH_INTERVAL_SECONDS)
    except WebSocketDisconnect:
        return


async def _read_snapshot(
    state: ServerState,
    *,
    run_dir: Path,
    metric_sequence: int | None,
    telemetry_sequence: int | None,
    log_stream: _LogStream | None,
) -> _result.Ok[_TrainingStreamSnapshot] | _result.Rejected:
    summary_result = await state.training_cli_client.summary(
        run_dir,
        metric_after=metric_sequence,
        telemetry_after=telemetry_sequence,
    )
    if isinstance(summary_result, _result.Rejected):
        return summary_result
    log_result = _read_log(run_dir, log_stream)
    if isinstance(log_result, _result.Rejected):
        return log_result
    return _result.Ok(
        value=_TrainingStreamSnapshot(
            summary=summary_result.value,
            log_stream=log_stream,
            log_content=log_result.value,
        )
    )


def _read_log(
    run_dir: Path,
    stream: _LogStream | None,
) -> _result.Ok[str | None] | _result.Rejected:
    if stream is None:
        return _result.Ok(value=None)
    result = read_log_tail(
        run_dir,
        stream=stream,
        max_bytes=_LOG_MAX_BYTES,
    )
    if isinstance(result, _result.Rejected):
        return result
    return _result.Ok(value=result.value)
