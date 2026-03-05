"""
main.py — API FastAPI para o projeto "Autorização Selfie".

Rotas:
  POST /verify   → recebe imagem (UploadFile) e retorna match + confiança
  GET  /users    → lista nomes carregados (debug)
  GET  /         → serve frontend estático (static/)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from face_logic import FaceService

# ------------------------------------------------------------------
# Instância global do serviço de reconhecimento facial
# ------------------------------------------------------------------
face_service: FaceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo e os embeddings assim que a API sobe."""
    global face_service  # noqa: PLW0603
    face_service = FaceService()
    face_service.load_from_disk()
    yield
    # Shutdown — nada a limpar


app = FastAPI(
    title="Autorização Selfie",
    version="0.1.0",
    lifespan=lifespan,
)

# ------------------------------------------------------------------
# CORS — libera tudo (POC)
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
    name, confidence = face_service.verify_user(contents)

    is_match = name not in ("Desconhecido", "Erro: imagem inválida", "Erro: nenhum rosto detectado")

    return JSONResponse(
        content={
            "match": is_match,
            "name": name,
            "confidence": confidence,
        }
    )


@app.get("/users")
async def list_users():
    """Retorna os nomes carregados da pasta database/ (debug)."""
    return {"users": list(face_service.known_faces.keys())}


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
