"""
config.py — Configurações centralizadas do microserviço.
"""

from pathlib import Path

# ── Diretórios ──────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATABASE_DIR = ROOT_DIR / "database"
LOGS_DIR = ROOT_DIR / "logs"
STATIC_DIR = ROOT_DIR / "static"

CACHE_FILE = DATA_DIR / "face_encodings.pkl"

# ── Modelo ──────────────────────────────────────────────────────────
MODEL_NAME = "buffalo_l"
MODEL_PROVIDERS = ["CPUExecutionProvider"]
DET_SIZE = (640, 640)

# ── Thresholds de confiança ─────────────────────────────────────────
THRESHOLD_MATCH = 0.65   # Match confirmado
THRESHOLD_DOUBT = 0.50   # Faixa de dúvida

# ── Imagens aceitas ─────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── API ─────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Identity Service — Reconhecimento Facial"
API_VERSION = "1.0.0"
