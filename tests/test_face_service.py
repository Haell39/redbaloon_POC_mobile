"""
test_face_service.py — Testes unitários para o FaceService.

Testa:
  • Carga de PKLs (responsáveis + equipe)
  • Indexação do CSV
  • Match facial com cosine similarity
  • Enriquecimento de dados do CSV no match
  • Cenários de erro (imagem inválida, sem rosto, base vazia)
  • Registro de novos rostos
  • Remoção de rostos
"""

import numpy as np
import pytest

from tests.conftest import (
    EMB_EQUIP_JULIA,
    EMB_EQUIP_RAFAEL,
    EMB_RESP_ADRIANA,
    EMB_RESP_CARLOS,
    make_embedding,
)


# ====================================================================
# PKL LOADING
# ====================================================================

class TestPKLLoading:
    """Testa se os PKLs são carregados corretamente."""

    def test_resp_bank_loaded(self, face_service):
        assert face_service.total_resp == 2

    def test_equip_bank_loaded(self, face_service):
        assert face_service.total_equip == 2

    def test_resp_has_correct_filenames(self, face_service):
        names = face_service.registered_names_resp
        assert "adriana-moura-padilha-1759115934569-8571.jpg" in names
        assert "carlos-silva-1234567890-1234.jpg" in names

    def test_equip_has_correct_filenames(self, face_service):
        names = face_service.registered_names_equip
        assert "rafael.jpg" in names
        assert "julia.jpg" in names

    def test_embedding_shape_is_512(self, face_service):
        entry = face_service.known_resp["adriana-moura-padilha-1759115934569-8571.jpg"]
        assert entry["embedding"].shape == (512,)

    def test_embedding_dtype_is_float32(self, face_service):
        entry = face_service.known_equip["rafael.jpg"]
        assert entry["embedding"].dtype == np.float32


# ====================================================================
# CSV INDEXING
# ====================================================================

class TestCSVIndexing:
    """Testa se o CSV foi indexado corretamente pela coluna 'foto'."""

    def test_csv_loaded(self, face_service):
        assert len(face_service.csv_index) == 2

    def test_csv_lookup_by_filename(self, face_service):
        row = face_service.csv_index.get("adriana-moura-padilha-1759115934569-8571.jpg")
        assert row is not None
        assert row["nome"] == "ADRIANA MOURA PADILHA"
        assert row["id"] == "2188"

    def test_csv_has_cpf(self, face_service):
        row = face_service.csv_index["carlos-silva-1234567890-1234.jpg"]
        assert row["cpf"] == "987.654.321-00"

    def test_equip_not_in_csv(self, face_service):
        """Equipe não deve ter entrada no CSV de responsáveis."""
        assert "rafael.jpg" not in face_service.csv_index


# ====================================================================
# COSINE SIMILARITY
# ====================================================================

class TestCosineSimilarity:
    """Testa a lógica de similaridade de cosseno."""

    def test_identical_vectors_score_1(self, face_service):
        emb = make_embedding(seed=42)
        score = face_service._cosine_similarity(emb, emb)
        assert abs(score - 1.0) < 1e-5

    def test_different_vectors_low_score(self, face_service):
        a = make_embedding(seed=1)
        b = make_embedding(seed=9999)
        score = face_service._cosine_similarity(a, b)
        assert score < 0.5

    def test_zero_vector_returns_zero(self, face_service):
        zero = np.zeros(512, dtype=np.float32)
        normal = make_embedding(seed=10)
        score = face_service._cosine_similarity(zero, normal)
        assert score == 0.0


# ====================================================================
# VERIFY — Match de Responsável
# ====================================================================

class TestVerifyResponsavel:
    """Testa verificação facial que retorna match de responsável."""

    def _make_fake_face(self, mock_insightface, embedding):
        fake_face = type("Face", (), {"embedding": embedding})()
        mock_insightface.get.return_value = [fake_face]

    def test_match_responsavel_returns_correct_status(self, face_service, mock_insightface):
        self._make_fake_face(mock_insightface, EMB_RESP_ADRIANA)
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        result = face_service.verify(img_bytes.tobytes())
        assert result["status"] == "match"
        assert result["confidence"] >= 0.99
        assert result["source"] == "resp"

    def test_match_responsavel_has_csv_data(self, face_service, mock_insightface):
        self._make_fake_face(mock_insightface, EMB_RESP_ADRIANA)
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        result = face_service.verify(img_bytes.tobytes())
        assert result["nome"] == "ADRIANA MOURA PADILHA"
        assert result["cpf"] == "123.456.789-00"
        assert result["id"] == "2188"

    def test_match_carlos_returns_correct_data(self, face_service, mock_insightface):
        self._make_fake_face(mock_insightface, EMB_RESP_CARLOS)
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        result = face_service.verify(img_bytes.tobytes())
        assert result["status"] == "match"
        assert result["nome"] == "CARLOS SILVA"
        assert result["id"] == "3001"


# ====================================================================
# VERIFY — Match de Equipe
# ====================================================================

class TestVerifyEquipe:

    def test_match_equipe_returns_correct_source(self, face_service, mock_insightface):
        fake_face = type("Face", (), {"embedding": EMB_EQUIP_RAFAEL})()
        mock_insightface.get.return_value = [fake_face]
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        result = face_service.verify(img_bytes.tobytes())
        assert result["status"] == "match"
        assert result["source"] == "equip"

    def test_match_equipe_no_csv_fields(self, face_service, mock_insightface):
        fake_face = type("Face", (), {"embedding": EMB_EQUIP_JULIA})()
        mock_insightface.get.return_value = [fake_face]
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        result = face_service.verify(img_bytes.tobytes())
        assert "nome" not in result


# ====================================================================
# VERIFY — No Match / Error
# ====================================================================

class TestVerifyErrors:

    def test_unknown_face_returns_no_match(self, face_service, mock_insightface):
        random_emb = make_embedding(seed=77777)
        fake_face = type("Face", (), {"embedding": random_emb})()
        mock_insightface.get.return_value = [fake_face]
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        result = face_service.verify(img_bytes.tobytes())
        assert result["status"] in ("no_match", "doubt")

    def test_no_face_detected(self, face_service, mock_insightface):
        mock_insightface.get.return_value = []
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        result = face_service.verify(img_bytes.tobytes())
        assert result["status"] == "error"

    def test_invalid_image_returns_error(self, face_service):
        result = face_service.verify(b"isto nao e uma imagem")
        assert result["status"] == "error"


# ====================================================================
# REGISTER
# ====================================================================

class TestRegister:
    """Testa cadastro de novos rostos via register()."""

    def test_register_success(self, face_service, mock_insightface):
        """Registra novo rosto com sucesso."""
        new_emb = make_embedding(seed=555)
        fake_face = type("Face", (), {"embedding": new_emb})()
        mock_insightface.get.return_value = [fake_face]
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        result = face_service.register("9999", img_bytes.tobytes())
        assert result["status"] == "ok"
        assert result["id"] == "9999"
        assert face_service.total_resp == 3  # 2 originais + 1 novo

    def test_register_adds_to_known_resp(self, face_service, mock_insightface):
        """O rosto registrado fica disponível pra verificação."""
        new_emb = make_embedding(seed=666)
        fake_face = type("Face", (), {"embedding": new_emb})()
        mock_insightface.get.return_value = [fake_face]
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        face_service.register("7777", img_bytes.tobytes())
        assert "7777.jpg" in face_service.known_resp

    def test_register_invalid_image(self, face_service):
        """Imagem inválida retorna erro."""
        result = face_service.register("1234", b"nao_e_imagem")
        assert result["status"] == "error"

    def test_register_no_face(self, face_service, mock_insightface):
        """Foto sem rosto retorna erro."""
        mock_insightface.get.return_value = []
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)
        result = face_service.register("1234", img_bytes.tobytes())
        assert result["status"] == "error"

    def test_register_overwrites_existing(self, face_service, mock_insightface):
        """Registrar com ID já existente sobrescreve o embedding."""
        new_emb = make_embedding(seed=888)
        fake_face = type("Face", (), {"embedding": new_emb})()
        mock_insightface.get.return_value = [fake_face]
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        face_service.register("5000", img_bytes.tobytes())
        face_service.register("5000", img_bytes.tobytes())
        assert face_service.total_resp == 3  # não duplica


# ====================================================================
# UNREGISTER
# ====================================================================

class TestUnregister:
    """Testa remoção de rostos."""

    def test_unregister_existing(self, face_service, mock_insightface):
        """Remove rosto previamente registrado."""
        new_emb = make_embedding(seed=777)
        fake_face = type("Face", (), {"embedding": new_emb})()
        mock_insightface.get.return_value = [fake_face]
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        face_service.register("to_remove", img_bytes.tobytes())
        assert face_service.total_resp == 3
        result = face_service.unregister("to_remove")
        assert result["status"] == "ok"
        assert face_service.total_resp == 2

    def test_unregister_nonexistent(self, face_service):
        """Remover ID inexistente retorna erro."""
        result = face_service.unregister("nao_existe")
        assert result["status"] == "error"


# ====================================================================
# REFRESH
# ====================================================================

class TestRefresh:

    def test_refresh_returns_counts(self, face_service):
        counts = face_service.refresh()
        assert counts["resp"] == 2
        assert counts["equip"] == 2
        assert counts["csv"] == 2

    def test_refresh_reloads_data(self, face_service):
        face_service.refresh()
        assert face_service.total_resp == 2
        assert face_service.total_equip == 2
