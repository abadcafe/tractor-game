"""Tractor game server — API layer."""

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Static files
_static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def index():
    html_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    return FileResponse(html_path)
