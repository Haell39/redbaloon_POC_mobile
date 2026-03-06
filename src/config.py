"""
config.py — Configurações centralizadas do microserviço.
"""

from pathlib import Path

# ── Diretórios ──────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT_DIR / "data"
LOGS_DIR  = ROOT_DIR / "logs"

# Banco de rostos — responsáveis (PKL gerado pelo Vetorizator)
DATABASE_DIR = ROOT_DIR / "database"

# Banco de rostos — equipe interna
DATABASE_EQUIP_DIR = ROOT_DIR / "database_equip"

# Tabela de responsáveis para enriquecimento de dados
CSV_FILE = ROOT_DIR / "tabelas" / "tabela_prod_responsaveis.csv"

# ── Modelo InsightFace ──────────────────────────────────────────────
MODEL_NAME      = "buffalo_l"
MODEL_PROVIDERS = ["CPUExecutionProvider"]
DET_SIZE        = (640, 640)

# ── Thresholds de confiança ─────────────────────────────────────────
THRESHOLD_MATCH = 0.65   # Match confirmado
THRESHOLD_DOUBT = 0.50   # Faixa de dúvida

# ── Imagens aceitas ─────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── PKL de registros dinâmicos (novos cadastros via /register) ──────
REGISTER_PKL_FILE = DATABASE_DIR / "registered_faces.pkl"

# ── API ─────────────────────────────────────────────────────────────
API_HOST    = "0.0.0.0"
API_PORT    = 8000
API_TITLE   = "Identity Service — Reconhecimento Facial"
API_VERSION = "1.2.0"

# ── Autenticação ────────────────────────────────────────────────────
import os
API_KEY = os.environ.get("API_KEY", "changeme-insecure-key-12345")
