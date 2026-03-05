"""
face_service.py — Serviço de reconhecimento facial (Microserviço de Identidade).

Responsabilidades:
  • Carregar embeddings do cache pickle (data/face_encodings.pkl).
  • Varrer pasta database/ e gerar embeddings quando cache não existe.
  • Comparar rosto recebido (bytes) via Similaridade de Cosseno.
  • Retornar contrato padronizado para integração com Node.js / Angular.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

from src.config import (
    CACHE_FILE,
    DATABASE_DIR,
    DET_SIZE,
    IMAGE_EXTENSIONS,
    MODEL_NAME,
    MODEL_PROVIDERS,
    THRESHOLD_DOUBT,
    THRESHOLD_MATCH,
)

logger = logging.getLogger("identity-service")


class FaceService:
    """Serviço de reconhecimento facial com cache em disco (pickle)."""

    def __init__(self) -> None:
        logger.info("Inicializando modelo InsightFace (%s / CPU)…", MODEL_NAME)
        self.app = FaceAnalysis(name=MODEL_NAME, providers=MODEL_PROVIDERS)
        self.app.prepare(ctx_id=0, det_size=DET_SIZE)
        self.known_faces: Dict[str, np.ndarray] = {}

    # ── Startup: cache → disco ──────────────────────────────────────
    def startup(self) -> None:
        """Tenta carregar do cache; se não existir, varre a pasta e salva."""
        if self._load_cache():
            return
        self.refresh()

    # ── Refresh forçado ─────────────────────────────────────────────
    def refresh(self) -> int:
        """Relê a pasta database/, gera embeddings e salva cache.

        Retorna a quantidade de rostos carregados.
        """
        self.known_faces.clear()
        self._scan_database()
        self._save_cache()
        return len(self.known_faces)

    # ── Verificação ─────────────────────────────────────────────────
    def verify(self, file_bytes: bytes) -> dict:
        """Recebe bytes de imagem e retorna resultado padronizado.

        Contrato de retorno (compatível Node.js):
            {
                "id":         str,     # nome do funcionário ou "unknown"
                "status":     str,     # "match" | "no_match" | "doubt" | "error"
                "confidence": float,   # 0.0 – 1.0
                "message":    str      # descrição legível
            }
        """
        # Decodifica bytes → imagem OpenCV
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Imagem recebida é inválida ou corrompida.")
            return self._response("unknown", "error", 0.0, "Imagem inválida ou corrompida.")

        # Detecção de rosto
        faces = self.app.get(img)
        if not faces:
            logger.warning("Nenhum rosto detectado na imagem enviada.")
            return self._response("unknown", "error", 0.0, "Nenhum rosto detectado na imagem.")

        # Base vazia
        if not self.known_faces:
            logger.warning("Tentativa de verificação com base de dados vazia.")
            return self._response("unknown", "error", 0.0, "Nenhum rosto cadastrado no sistema.")

        # Comparação por Cosine Similarity
        query_emb = faces[0].embedding
        best_name, best_score = self._find_best_match(query_emb)

        if best_score >= THRESHOLD_MATCH:
            logger.info("Match: %s (%.4f)", best_name, best_score)
            return self._response(best_name, "match", best_score, "Acesso autorizado.")

        if best_score >= THRESHOLD_DOUBT:
            logger.info("Dúvida: %s (%.4f)", best_name, best_score)
            return self._response(
                best_name, "doubt", best_score,
                "Confiança insuficiente. Tente novamente com melhor iluminação.",
            )

        logger.info("No match (%.4f)", best_score)
        return self._response("unknown", "no_match", best_score, "Rosto não reconhecido.")

    # ── Propriedades públicas ───────────────────────────────────────
    @property
    def registered_names(self) -> list[str]:
        return sorted(self.known_faces.keys())

    @property
    def total_registered(self) -> int:
        return len(self.known_faces)

    # ================================================================
    #  INTERNALS
    # ================================================================

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """similarity = (A · B) / (‖A‖ × ‖B‖)"""
        return float(np.dot(a, b) / (norm(a) * norm(b)))

    @staticmethod
    def _response(id_: str, status: str, confidence: float, message: str) -> dict:
        return {
            "id": id_,
            "status": status,
            "confidence": round(confidence, 4),
            "message": message,
        }

    def _find_best_match(self, query_emb: np.ndarray) -> Tuple[str, float]:
        best_name = "unknown"
        best_score = 0.0
        for name, known_emb in self.known_faces.items():
            score = self._cosine_similarity(query_emb, known_emb)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name, best_score

    # ── Scan database/ ──────────────────────────────────────────────
    def _scan_database(self) -> None:
        if not DATABASE_DIR.exists():
            logger.warning("Pasta '%s' não encontrada.", DATABASE_DIR)
            return

        logger.info("Varrendo pasta '%s'…", DATABASE_DIR)
        for file in sorted(DATABASE_DIR.iterdir()):
            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                img = cv2.imread(str(file))
                if img is None:
                    logger.warning("Não foi possível ler: %s", file.name)
                    continue

                faces = self.app.get(img)
                if not faces:
                    logger.warning("Nenhum rosto detectado em: %s", file.name)
                    continue

                self.known_faces[file.stem] = faces[0].embedding
                logger.info("  Cadastrado: %s", file.stem)
            except Exception:
                logger.exception("Erro ao processar '%s'", file.name)

        logger.info("%d rosto(s) carregado(s).", len(self.known_faces))

    # ── Pickle: salvar ──────────────────────────────────────────────
    def _save_cache(self) -> None:
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {name: emb.tolist() for name, emb in self.known_faces.items()}
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Cache salvo: %s (%d rosto(s)).", CACHE_FILE, len(data))
        except Exception:
            logger.exception("Falha ao salvar cache.")

    # ── Pickle: carregar ────────────────────────────────────────────
    def _load_cache(self) -> bool:
        """Retorna True se carregou o cache com sucesso."""
        if not CACHE_FILE.exists():
            logger.info("Cache não encontrado. Será gerado agora.")
            return False
        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)  # noqa: S301

            if not isinstance(data, dict) or not data:
                logger.warning("Cache vazio ou formato inesperado. Regenerando…")
                return False

            self.known_faces = {
                name: np.array(emb, dtype=np.float32)
                for name, emb in data.items()
            }
            logger.info("Cache carregado: %d rosto(s) (load instantâneo).", len(self.known_faces))
            return True

        except (pickle.UnpicklingError, EOFError, ValueError) as exc:
            logger.warning("Cache corrompido (%s). Regenerando…", exc)
            return False
        except Exception:
            logger.exception("Erro inesperado ao ler cache. Regenerando…")
            return False
