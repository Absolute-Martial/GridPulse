"""FastAPI entrypoint for the GridPulse backend."""

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Please use `import python_multipart` instead\.",
    category=PendingDeprecationWarning,
    module=r"starlette\.formparsers",
)

from fastapi import FastAPI

from app.api.router import api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="GridPulse Backend",
        docs_url=None,
        redoc_url=None,
    )
    app.include_router(api_router)
    return app


app = create_app()
