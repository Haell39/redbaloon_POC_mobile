"""
conftest.py — Fixtures compartilhadas para todos os testes.

Mockamos o InsightFace completamente para que os testes rodem
instantaneamente, sem baixar modelos nem precisar de GPU.
Patcheamos os paths diretamente no módulo face_service para
que usem diretórios temporários.
"""

import csv
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Garante que o código-fonte está no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Embeddings fake para testes ─────────────────────────────────────
def make_embedding(seed: int = 0) -> np.ndarray:
    """Gera um vetor 512-d normalizado (cosine-safe) reprodutível."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(512).astype(np.float32)
    emb /= np.linalg.norm(emb)
    return emb


# Embeddings fixos para responsáveis e equipe
EMB_RESP_ADRIANA  = make_embedding(seed=42)
EMB_RESP_CARLOS   = make_embedding(seed=99)
EMB_EQUIP_RAFAEL  = make_embedding(seed=200)
EMB_EQUIP_JULIA   = make_embedding(seed=201)


def _create_test_data(tmp_path: Path) -> Path:
    """Cria estrutura de pastas temporária com PKLs e CSV de teste."""

    # ── PKL de responsáveis ─────────────────────────────────────────
    db_dir = tmp_path / "database"
    db_dir.mkdir(exist_ok=True)
    resp_data = [
        {
            "id": "adriana-moura-padilha-1759115934569-8571",
            "filename": "adriana-moura-padilha-1759115934569-8571.jpg",
            "embedding": EMB_RESP_ADRIANA,
        },
        {
            "id": "carlos-silva-1234567890-1234",
            "filename": "carlos-silva-1234567890-1234.jpg",
            "embedding": EMB_RESP_CARLOS,
        },
    ]
    with open(db_dir / "face_encodings_001.pkl", "wb") as f:
        pickle.dump(resp_data, f)

    # ── PKL de equipe ───────────────────────────────────────────────
    equip_dir = tmp_path / "database_equip"
    equip_dir.mkdir(exist_ok=True)
    equip_data = [
        {
            "id": "rafael",
            "filename": "rafael.jpg",
            "embedding": EMB_EQUIP_RAFAEL,
        },
        {
            "id": "julia",
            "filename": "julia.jpg",
            "embedding": EMB_EQUIP_JULIA,
        },
    ]
    with open(equip_dir / "face_encodings.pkl", "wb") as f:
        pickle.dump(equip_data, f)

    # ── CSV de responsáveis ─────────────────────────────────────────
    tabelas_dir = tmp_path / "tabelas"
    tabelas_dir.mkdir(exist_ok=True)
    csv_path = tabelas_dir / "tabela de responsaveishomolog.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "nome", "numero", "foto", "cpf", "ativo", "origem"],
        )
        writer.writeheader()
        writer.writerow({
            "id": "2188",
            "nome": "ADRIANA MOURA PADILHA",
            "numero": "11999999999",
            "foto": "adriana-moura-padilha-1759115934569-8571.jpg",
            "cpf": "123.456.789-00",
            "ativo": "true",
            "origem": "HOMOLOGAÇÃO",
        })
        writer.writerow({
            "id": "3001",
            "nome": "CARLOS SILVA",
            "numero": "21988888888",
            "foto": "carlos-silva-1234567890-1234.jpg",
            "cpf": "987.654.321-00",
            "ativo": "true",
            "origem": "HOMOLOGAÇÃO",
        })

    # ── Diretórios extras ───────────────────────────────────────────
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "data").mkdir(exist_ok=True)

    return tmp_path


@pytest.fixture
def tmp_project(tmp_path):
    """Cria estrutura de pastas temporária com PKLs e CSV de teste."""
    return _create_test_data(tmp_path)


@pytest.fixture
def mock_insightface():
    """Mocka o InsightFace para não carregar modelo real."""
    with patch("src.services.face_service.FaceAnalysis") as MockFA:
        mock_app = MagicMock()
        mock_app.prepare.return_value = None
        mock_app.get.return_value = []  # Padrão: nenhum rosto detectado
        MockFA.return_value = mock_app
        yield mock_app


@pytest.fixture
def face_service(tmp_project, mock_insightface):
    """FaceService totalmente funcional contra dados de teste.

    Patcha os paths diretamente no módulo face_service (onde eles são
    importados como constantes) para usar os diretórios temporários.
    """
    with patch.multiple(
        "src.services.face_service",
        DATABASE_DIR=tmp_project / "database",
        DATABASE_EQUIP_DIR=tmp_project / "database_equip",
        CSV_FILE=tmp_project / "tabelas" / "tabela de responsaveishomolog.csv",
        REGISTER_PKL_FILE=tmp_project / "database" / "registered_faces.pkl",
    ):
        from src.services.face_service import FaceService
        svc = FaceService()
        svc.startup()
        yield svc
