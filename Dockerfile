# ── Stage base ───────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Sem buffer + sem .pyc + locale consistente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependências de sistema para OpenCV + InsightFace
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependências Python (camada cacheável) ───────────────────────────
COPY requirements.txt .
RUN pip install -r requirements.txt

# ── Código-fonte ─────────────────────────────────────────────────────
COPY src/ ./src/

# ── Dados (PKLs e tabelas) ───────────────────────────────────────────
# Criados aqui para garantir que existam mesmo sem volume mapeado
RUN mkdir -p database tabelas logs data

COPY database/      ./database/
COPY tabelas/       ./tabelas/

# ── Porta exposta ────────────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck (EasyPanel aguarda "healthy" antes de rotear tráfego) ─
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=120s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Entrypoint ───────────────────────────────────────────────────────
# workers=1 para CPU única do KVM 4; aumente se tiver mais vCPUs
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
