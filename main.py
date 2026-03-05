"""
main.py — API FastAPI para o projeto "Autorização Selfie" (MVP).

Rotas:
  POST /verify      → recebe selfie (UploadFile) e retorna verificação facial
  GET  /users       → lista nomes cadastrados (debug)
  GET  /refresh-db  → força releitura da pasta database/ e recria cache
  GET  /             → serve frontend estático (static/)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from face_core import FaceService

# ------------------------------------------------------------------
# Instância global do serviço de reconhecimento facial
# ------------------------------------------------------------------
face_service: FaceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo e os embeddings (cache ou disco) no startup."""
    global face_service  # noqa: PLW0603
    face_service = FaceService()
    face_service.startup()
    yield


app = FastAPI(
    title="Autorização Selfie",
    version="0.2.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# CORS — libera tudo (MVP local, sem HTTPS)
# ------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Rotas
# ------------------------------------------------------------------
@app.post("/verify")
async def verify(file: UploadFile = File(...)):
    """Recebe uma selfie e retorna o resultado da verificação."""
    contents = await file.read()
    result = face_service.verify_user(contents)
    return JSONResponse(content=result)


@app.get("/users")
async def list_users():
    """Retorna os nomes carregados da pasta database/ (debug)."""
    return {"users": face_service.registered_names}


@app.get("/refresh-db")
async def refresh_database():
    """Força a releitura da pasta database/ e recria o cache pkl."""
    total = face_service.refresh()
    return {
        "status": "ok",
        "message": f"Base atualizada com {total} rosto(s).",
        "users": face_service.registered_names,
    }


# ------------------------------------------------------------------
# Servir arquivos estáticos (frontend) na raiz
# ------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
