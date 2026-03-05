"""
main.py — Microserviço de Identidade (FastAPI).

Endpoints:
  POST /verify      → recebe imagem e retorna verificação facial
  GET  /refresh-db  → força releitura da pasta database/ e recria cache
  GET  /users       → lista nomes cadastrados (debug)
  GET  /health      → healthcheck para orquestradores
  GET  /            → serve test client estático (static/)
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import API_HOST, API_PORT, API_TITLE, API_VERSION, LOGS_DIR, STATIC_DIR
from src.services.face_service import FaceService

# ── Logging ─────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "service.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("identity-service")

# ── Service singleton ───────────────────────────────────────────────
face_service: FaceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega modelo e embeddings no startup."""
    global face_service  # noqa: PLW0603
    logger.info("Iniciando Identity Service…")
    face_service = FaceService()
    face_service.startup()
    logger.info(
        "Pronto — %d rosto(s) carregado(s).", face_service.total_registered
    )
    yield
    logger.info("Encerrando Identity Service.")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan,
)

# ── CORS — libera qualquer origem (Angular 4200 → Python 8000) ─────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================================================================
# ENDPOINTS
# ====================================================================

@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    """Recebe selfie (multipart) e retorna verificação facial.

    Retorno:
        {"id": str, "status": "match"|"no_match"|"doubt"|"error", "confidence": float, "message": str}
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Arquivo enviado está vazio.")

    result = face_service.verify(contents)
    return JSONResponse(content=result)


@app.get("/refresh-db")
async def refresh_database():
    """Força releitura da pasta database/ e recria o cache."""
    total = face_service.refresh()
    logger.info("Base atualizada: %d rosto(s).", total)
    return {
        "status": "ok",
        "message": f"Base atualizada com {total} rosto(s).",
        "users": face_service.registered_names,
    }


@app.get("/users")
async def list_users():
    """Lista nomes cadastrados (debug)."""
    return {
        "total": face_service.total_registered,
        "users": face_service.registered_names,
    }


@app.get("/health")
async def health():
    """Healthcheck para Docker / orquestradores."""
    return {
        "status": "healthy",
        "model": "buffalo_l",
        "registered_faces": face_service.total_registered,
    }


# ── Arquivos estáticos (test client) ───────────────────────────────
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


# ── Entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
