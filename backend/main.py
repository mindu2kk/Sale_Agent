"""Compatibility entrypoint for the FastAPI app.

The application lives in ``backend.api.main``. This module keeps older
commands such as ``uvicorn backend.main:app`` and existing tests working.
"""

from backend.api.main import *  # noqa: F401,F403


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
