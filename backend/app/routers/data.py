"""Dataset import/export (CSV, Excel, pickle) per project step."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.config import get_settings
from app.services import dataset_io, project_service

router = APIRouter(prefix="/projects", tags=["datasets"])

_MAX_BYTES = 50 * 1024 * 1024


async def _save_upload_temp(upload: UploadFile) -> Path:
    data = await upload.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(413, "File too large (max 50MB)")
    suffix = Path(upload.filename or "upload").suffix.lower() or ".csv"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.flush()
        return Path(tmp.name)
    finally:
        tmp.close()


@router.get("/{project_id}/data/last-run")
def last_run_manifest(project_id: str) -> dict:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    d = dataset_io.last_run_dir(project_id)
    if not d.exists():
        return {"path": None, "files": []}
    files = sorted(f.name for f in d.iterdir() if f.is_file())
    root = get_settings().data_dir
    try:
        rel = str(d.relative_to(root))
    except ValueError:
        rel = str(d)
    return {"path": rel, "files": files}


@router.post("/{project_id}/data/import/articles")
async def import_articles(project_id: str, file: UploadFile = File(...)) -> dict:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    path = await _save_upload_temp(file)
    try:
        _df, rows, stats = dataset_io.read_articles_from_path(path)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        path.unlink(missing_ok=True)
    return {
        "kind": "articles",
        "row_count": len(rows),
        "articles": rows,
        "import_stats": stats,
    }


@router.post("/{project_id}/data/import/mentions")
async def import_mentions(project_id: str, file: UploadFile = File(...)) -> dict:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    path = await _save_upload_temp(file)
    try:
        _df, rows = dataset_io.read_mentions_from_path(path)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        path.unlink(missing_ok=True)
    return {"kind": "mentions", "row_count": len(rows), "mentions": rows}


Artifact = Literal["classification", "mentions", "normalized", "bundle"]
FormatParam = Literal["csv", "xlsx", "pkl"]


@router.get("/{project_id}/data/export/{artifact}")
def export_artifact(
    project_id: str,
    artifact: Artifact,
    file_format: FormatParam = Query("csv", alias="format"),
) -> Response:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")

    if artifact == "bundle":
        if file_format not in ("pkl", "xlsx"):
            raise HTTPException(
                400, "Bundle export supports format=pkl or format=xlsx only"
            )
        try:
            body, media, fname = dataset_io.export_bundle_bytes(project_id, file_format)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        return Response(
            content=body,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        )

    try:
        df = dataset_io.load_artifact_dataframe(project_id, artifact)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e

    body, media = dataset_io.export_dataframe_bytes(df, file_format, sheet_name=artifact)
    ext = {"csv": "csv", "xlsx": "xlsx", "pkl": "pkl"}[file_format]
    fname = f"l2g_{artifact}.{ext}"
    return Response(
        content=body,
        media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@router.get("/{project_id}/data/templates/articles")
def template_articles(project_id: str) -> Response:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    body = dataset_io.articles_template_csv_bytes()
    return Response(
        content=body,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="articles_template.csv"'
        },
    )


@router.get("/{project_id}/data/templates/mentions")
def template_mentions(project_id: str) -> Response:
    if not project_service.get_project(project_id):
        raise HTTPException(404, "Project not found")
    body = dataset_io.mentions_template_csv_bytes()
    return Response(
        content=body,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="mentions_template.csv"'
        },
    )
