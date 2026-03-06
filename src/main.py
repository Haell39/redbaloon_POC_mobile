"""
main.py — Microserviço de Identidade (FastAPI).

Endpoints:
  POST /verify      → recebe imagem e retorna verificação facial (dual-bank + CSV)
  POST /register    → recebe id + foto → vetoriza e cadastra no banco de responsáveis
  DELETE /unregister/{id} → remove rosto do banco de responsáveis
  GET  /refresh-db  → recarrega PKLs e CSV sem reiniciar o serviço
  GET  /users       → lista rostos cadastrados em cada banco (debug)
  GET  /health      → healthcheck para Docker / EasyPanel

Autenticação:
  Todas as rotas (exceto /health) exigem header X-API-Key.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from src.config import API_HOST, API_KEY, API_PORT, API_TITLE, API_VERSION, LOGS_DIR
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
        "Pronto — %d responsável(is), %d membro(s) de equipe.",
        face_service.total_resp,
        face_service.total_equip,
    )
    yield
    logger.info("Encerrando Identity Service.")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Autenticação por API Key ────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Depends(api_key_header)):
    """Valida X-API-Key em todas as rotas protegidas."""
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida ou ausente.")
    return api_key


# ====================================================================
# ENDPOINTS
# ====================================================================

@app.post("/verify", dependencies=[Depends(require_api_key)])
async def verify(file: UploadFile = File(...)):
    """Recebe foto (multipart) e retorna verificação facial.

    Retorno base:
        {"id", "filename", "status", "confidence", "source", "message"}
    Campos adicionais quando source=="resp" e há dados no CSV:
        {"nome", "cpf", "numero", "ativo", "origem"}
    """
    if face_service is None:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Arquivo enviado está vazio.")

    result = face_service.verify(contents)
    return JSONResponse(content=result)


@app.post("/register", dependencies=[Depends(require_api_key)])
async def register(
    id: str = Form(...),
    file: UploadFile = File(...),
):
    """Recebe id do responsável + foto → vetoriza e cadastra no banco.

    O Node.js envia:
        - id:   ID numérico do responsável (ex: "2188")
        - file: Foto do rosto (JPEG/PNG)

    O serviço vetoriza a foto em tempo real e adiciona ao banco de responsáveis.
    """
    if face_service is None:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Arquivo enviado está vazio.")

    result = face_service.register(resp_id=id, file_bytes=contents)
    status_code = 200 if result["status"] == "ok" else 400
    return JSONResponse(content=result, status_code=status_code)


@app.delete("/unregister/{resp_id}", dependencies=[Depends(require_api_key)])
async def unregister(resp_id: str):
    """Remove um rosto do banco de responsáveis pelo ID."""
    if face_service is None:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    result = face_service.unregister(resp_id)
    status_code = 200 if result["status"] == "ok" else 404
    return JSONResponse(content=result, status_code=status_code)


@app.get("/refresh-db", dependencies=[Depends(require_api_key)])
async def refresh_database():
    """Recarrega PKLs dos dois bancos e o índice do CSV sem reiniciar."""
    if face_service is None:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    counts = face_service.refresh()
    logger.info("Base recarregada: %s", counts)
    return {
        "status":  "ok",
        "message": (
            f"Base atualizada — {counts['resp']} responsável(is), "
            f"{counts['equip']} equipe, {counts['csv']} entrada(s) no CSV."
        ),
        "resp_total":  counts["resp"],
        "equip_total": counts["equip"],
        "csv_total":   counts["csv"],
    }


@app.get("/users", dependencies=[Depends(require_api_key)])
async def list_users():
    """Lista rostos cadastrados em cada banco (debug)."""
    if face_service is None:
        raise HTTPException(status_code=503, detail="Serviço não inicializado.")

    return {
        "resp": {
            "total": face_service.total_resp,
            "users": face_service.registered_names_resp,
        },
        "equip": {
            "total": face_service.total_equip,
            "users": face_service.registered_names_equip,
        },
    }


@app.get("/health")
async def health():
    """Healthcheck para Docker / EasyPanel. NÃO exige API Key."""
    if face_service is None:
        return JSONResponse(status_code=503, content={"status": "starting"})

    return {
        "status":      "healthy",
        "model":       "buffalo_l",
        "resp_faces":  face_service.total_resp,
        "equip_faces": face_service.total_equip,
    }


# ── Entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
