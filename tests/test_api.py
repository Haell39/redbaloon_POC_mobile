"""
test_api.py — Testes dos endpoints HTTP (API Key + Register + Unregister).
"""

import os
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import EMB_RESP_ADRIANA, make_embedding

TEST_API_KEY = "test-key-12345"


@pytest.fixture
def client(tmp_project, mock_insightface):
    """TestClient da FastAPI com paths patcheados e API Key configurada."""
    patches = {
        "DATABASE_DIR": tmp_project / "database",
        "DATABASE_EQUIP_DIR": tmp_project / "database_equip",
        "CSV_FILE": tmp_project / "tabelas" / "tabela de responsaveishomolog.csv",
        "REGISTER_PKL_FILE": tmp_project / "database" / "registered_faces.pkl",
    }

    with patch.multiple("src.services.face_service", **patches), \
         patch.multiple("src.config", **patches), \
         patch("src.config.API_KEY", TEST_API_KEY), \
         patch("src.main.API_KEY", TEST_API_KEY):
        from src.main import app
        with TestClient(app) as c:
            yield c, mock_insightface


def auth_headers():
    return {"X-API-Key": TEST_API_KEY}


# ====================================================================
# API KEY
# ====================================================================

class TestApiKey:

    def test_no_api_key_returns_401(self, client):
        c, _ = client
        resp = c.get("/users")
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, client):
        c, _ = client
        resp = c.get("/users", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_correct_api_key_returns_200(self, client):
        c, _ = client
        resp = c.get("/users", headers=auth_headers())
        assert resp.status_code == 200

    def test_health_does_not_require_api_key(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200


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
        resp = c.get("/users", headers=auth_headers())
        assert resp.status_code == 200

    def test_users_separates_banks(self, client):
        c, _ = client
        data = c.get("/users", headers=auth_headers()).json()
        assert data["resp"]["total"] == 2
        assert data["equip"]["total"] == 2


# ====================================================================
# REFRESH-DB
# ====================================================================

class TestRefreshEndpoint:

    def test_refresh_returns_ok(self, client):
        c, _ = client
        resp = c.get("/refresh-db", headers=auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


# ====================================================================
# VERIFY
# ====================================================================

class TestVerifyEndpoint:

    def test_verify_requires_api_key(self, client):
        c, _ = client
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        resp = c.post("/verify", files={"file": ("s.jpg", img_bytes.tobytes(), "image/jpeg")})
        assert resp.status_code == 401

    def test_verify_with_api_key(self, client):
        c, mock_app = client
        import cv2
        fake_face = type("Face", (), {"embedding": EMB_RESP_ADRIANA})()
        mock_app.get.return_value = [fake_face]
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        resp = c.post(
            "/verify",
            files={"file": ("s.jpg", img_bytes.tobytes(), "image/jpeg")},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "match"


# ====================================================================
# REGISTER
# ====================================================================

class TestRegisterEndpoint:

    def test_register_requires_api_key(self, client):
        c, _ = client
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        resp = c.post(
            "/register",
            data={"id": "999"},
            files={"file": ("f.jpg", img_bytes.tobytes(), "image/jpeg")},
        )
        assert resp.status_code == 401

    def test_register_success(self, client):
        c, mock_app = client
        import cv2
        new_emb = make_embedding(seed=555)
        fake_face = type("Face", (), {"embedding": new_emb})()
        mock_app.get.return_value = [fake_face]
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        resp = c.post(
            "/register",
            data={"id": "999"},
            files={"file": ("f.jpg", img_bytes.tobytes(), "image/jpeg")},
            headers=auth_headers(),
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["id"] == "999"

    def test_register_no_face_returns_400(self, client):
        c, mock_app = client
        import cv2
        mock_app.get.return_value = []
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        resp = c.post(
            "/register",
            data={"id": "999"},
            files={"file": ("f.jpg", img_bytes.tobytes(), "image/jpeg")},
            headers=auth_headers(),
        )
        assert resp.status_code == 400


# ====================================================================
# UNREGISTER
# ====================================================================

class TestUnregisterEndpoint:

    def test_unregister_requires_api_key(self, client):
        c, _ = client
        resp = c.delete("/unregister/9999")
        assert resp.status_code == 401

    def test_unregister_nonexistent_returns_404(self, client):
        c, _ = client
        resp = c.delete("/unregister/nao_existe", headers=auth_headers())
        assert resp.status_code == 404

    def test_unregister_after_register(self, client):
        c, mock_app = client
        import cv2
        new_emb = make_embedding(seed=555)
        fake_face = type("Face", (), {"embedding": new_emb})()
        mock_app.get.return_value = [fake_face]
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        c.post(
            "/register",
            data={"id": "to_delete"},
            files={"file": ("f.jpg", img_bytes.tobytes(), "image/jpeg")},
            headers=auth_headers(),
        )
        resp = c.delete("/unregister/to_delete", headers=auth_headers())
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
