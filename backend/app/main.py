"""L2G FastAPI entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import data, external_data, pipeline, projects, system, training

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="L2G: disease-agnostic literature mining API (BioBERT relevance, NER, normalization).",
)
print("Loaded CORS origins:", settings.cors_origins)
print("CORS allow_all:", settings.cors_allow_all, "allow_credentials:", settings.cors_allow_credentials)

allow_origins = ["*"] if settings.cors_allow_all else settings.cors_origins
allow_credentials = False if settings.cors_allow_all else settings.cors_allow_credentials

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(projects.router)
app.include_router(data.router)
app.include_router(external_data.router)
app.include_router(training.router)
app.include_router(pipeline.router)


@app.get("/")
def root() -> dict:
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": "/health",
    }
