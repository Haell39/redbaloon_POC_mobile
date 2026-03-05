"""
test_api.py — Testes dos endpoints HTTP da API.

Testa as rotas /health, /users, /refresh-db e /verify via TestClient.
O InsightFace é mockado e os paths são patcheados para dados temporários.
"""

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import EMB_RESP_ADRIANA, make_embedding


@pytest.fixture
def client(tmp_project, mock_insightface):
    """TestClient da FastAPI com paths patcheados para dados temporários."""
    patches = {
        "DATABASE_DIR": tmp_project / "database",
        "DATABASE_EQUIP_DIR": tmp_project / "database_equip",
        "CSV_FILE": tmp_project / "tabelas" / "tabela de responsaveishomolog.csv",
    }

    with patch.multiple("src.services.face_service", **patches), \
         patch.multiple("src.config", **patches):
        from src.main import app
        with TestClient(app) as c:
            yield c, mock_insightface


# ====================================================================
# HEALTH
# ====================================================================

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200

    def test_health_shows_both_banks(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert data["status"] == "healthy"
        assert data["resp_faces"] == 2
        assert data["equip_faces"] == 2


# ====================================================================
# USERS
# ====================================================================

class TestUsersEndpoint:

    def test_users_returns_200(self, client):
        c, _ = client
        resp = c.get("/users")
        assert resp.status_code == 200

    def test_users_separates_banks(self, client):
        c, _ = client
        data = c.get("/users").json()
        assert data["resp"]["total"] == 2
        assert data["equip"]["total"] == 2
        assert "adriana-moura-padilha-1759115934569-8571.jpg" in data["resp"]["users"]
        assert "rafael.jpg" in data["equip"]["users"]


# ====================================================================
# REFRESH-DB
# ====================================================================

class TestRefreshEndpoint:

    def test_refresh_returns_ok(self, client):
        c, _ = client
        resp = c.get("/refresh-db")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["resp_total"] == 2
        assert data["equip_total"] == 2
        assert data["csv_total"] == 2


# ====================================================================
# VERIFY
# ====================================================================

class TestVerifyEndpoint:

    def test_verify_empty_file_returns_400(self, client):
        c, _ = client
        resp = c.post("/verify", files={"file": ("test.jpg", b"", "image/jpeg")})
        assert resp.status_code == 400

    def test_verify_with_valid_face_match(self, client):
        """Envia selfie com embedding idêntico → match de responsável."""
        c, mock_app = client
        import cv2

        fake_face = type("Face", (), {"embedding": EMB_RESP_ADRIANA})()
        mock_app.get.return_value = [fake_face]

        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        resp = c.post("/verify", files={"file": ("selfie.jpg", img_bytes.tobytes(), "image/jpeg")})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "match"
        assert data["source"] == "resp"
        assert data["nome"] == "ADRIANA MOURA PADILHA"

    def test_verify_no_face_detected(self, client):
        """Imagem sem rosto retorna error."""
        c, mock_app = client
        import cv2

        mock_app.get.return_value = []
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        resp = c.post("/verify", files={"file": ("selfie.jpg", img_bytes.tobytes(), "image/jpeg")})

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
