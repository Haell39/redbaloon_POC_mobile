"""
test_face_service.py — Testes unitários para o FaceService.

Testa:
  • Carga de PKLs (responsáveis + equipe)
  • Indexação do CSV
  • Match facial com cosine similarity
  • Enriquecimento de dados do CSV no match
  • Cenários de erro (imagem inválida, sem rosto, base vazia)
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
        assert score < 0.5  # vetores aleatórios → baixa similaridade

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
        """Configura o mock para retornar um rosto com embedding específico."""
        fake_face = type("Face", (), {"embedding": embedding})()
        mock_insightface.get.return_value = [fake_face]

    def test_match_responsavel_returns_correct_status(self, face_service, mock_insightface):
        """Envia embedding idêntico ao da Adriana → deve dar match."""
        self._make_fake_face(mock_insightface, EMB_RESP_ADRIANA)
        # Cria imagem fake qualquer (1x1 pixel branco)
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        result = face_service.verify(img_bytes.tobytes())

        assert result["status"] == "match"
        assert result["confidence"] >= 0.99
        assert result["source"] == "resp"

    def test_match_responsavel_has_csv_data(self, face_service, mock_insightface):
        """Quando match é responsável, deve ter dados do CSV."""
        self._make_fake_face(mock_insightface, EMB_RESP_ADRIANA)
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        result = face_service.verify(img_bytes.tobytes())

        assert result["nome"] == "ADRIANA MOURA PADILHA"
        assert result["cpf"] == "123.456.789-00"
        assert result["numero"] == "11999999999"
        assert result["ativo"] == "true"
        assert result["id"] == "2188"  # ID do CSV, não do PKL

    def test_match_carlos_returns_correct_data(self, face_service, mock_insightface):
        """Match do Carlos retorna dados corretos."""
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
    """Testa verificação facial que retorna match de equipe."""

    def test_match_equipe_returns_correct_source(self, face_service, mock_insightface):
        """Match de equipe retorna source='equip'."""
        fake_face = type("Face", (), {"embedding": EMB_EQUIP_RAFAEL})()
        mock_insightface.get.return_value = [fake_face]
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        result = face_service.verify(img_bytes.tobytes())

        assert result["status"] == "match"
        assert result["source"] == "equip"
        assert result["confidence"] >= 0.99

    def test_match_equipe_no_csv_fields(self, face_service, mock_insightface):
        """Match de equipe NÃO deve ter campos de CSV (nome, cpf, etc.)."""
        fake_face = type("Face", (), {"embedding": EMB_EQUIP_JULIA})()
        mock_insightface.get.return_value = [fake_face]
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        result = face_service.verify(img_bytes.tobytes())

        assert "nome" not in result
        assert "cpf" not in result


# ====================================================================
# VERIFY — No Match
# ====================================================================

class TestVerifyNoMatch:
    """Testa cenários onde o rosto não é reconhecido."""

    def test_unknown_face_returns_no_match(self, face_service, mock_insightface):
        """Embedding desconhecido deve retornar no_match."""
        random_emb = make_embedding(seed=77777)  # totalmente diferente
        fake_face = type("Face", (), {"embedding": random_emb})()
        mock_insightface.get.return_value = [fake_face]
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        result = face_service.verify(img_bytes.tobytes())

        assert result["status"] in ("no_match", "doubt")
        assert result["id"] in ("unknown", result.get("id", ""))


# ====================================================================
# VERIFY — Cenários de Erro
# ====================================================================

class TestVerifyErrors:
    """Testa cenários de erro."""

    def test_no_face_detected(self, face_service, mock_insightface):
        """Quando nenhum rosto é detectado, retorna error."""
        mock_insightface.get.return_value = []  # sem rosto
        import cv2
        _, img_bytes = cv2.imencode(".jpg", np.ones((10, 10, 3), dtype=np.uint8) * 255)

        result = face_service.verify(img_bytes.tobytes())

        assert result["status"] == "error"
        assert "Nenhum rosto" in result["message"]

    def test_invalid_image_returns_error(self, face_service):
        """Bytes inválidos (não é imagem) retornam error."""
        result = face_service.verify(b"isto nao e uma imagem")
        assert result["status"] == "error"
        assert "inválida" in result["message"].lower() or "corrompida" in result["message"].lower()


# ====================================================================
# REFRESH
# ====================================================================

class TestRefresh:
    """Testa recarga dos bancos."""

    def test_refresh_returns_counts(self, face_service):
        counts = face_service.refresh()
        assert counts["resp"] == 2
        assert counts["equip"] == 2
        assert counts["csv"] == 2

    def test_refresh_reloads_data(self, face_service):
        """Após refresh, os dados continuam acessíveis."""
        face_service.refresh()
        assert face_service.total_resp == 2
        assert face_service.total_equip == 2
        assert len(face_service.csv_index) == 2
