"""
face_core.py — Serviço de reconhecimento facial (MVP).

Evolução da POC:
  • Persistência de embeddings via pickle (cache em disco).
  • Load instantâneo quando `face_encodings.pkl` já existe.
  • Refresh sob demanda (endpoint `/refresh-db`).
  • Três faixas de confiança: Match (>0.65), Dúvida (0.50–0.65), Desconhecido (<0.50).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# ── Constantes ──────────────────────────────────────────────────────
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = _BASE_DIR / "database"
CACHE_FILE = _BASE_DIR / "face_encodings.pkl"

# Thresholds de confiança
THRESHOLD_MATCH = 0.65   # Verde  — acesso autorizado
THRESHOLD_DOUBT = 0.50   # Amarelo — tente novamente


class FaceService:
    """Serviço de reconhecimento facial com cache em disco (pickle)."""

    def __init__(self) -> None:
        print("[FaceService] Inicializando modelo InsightFace (buffalo_l / CPU)…")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces: Dict[str, np.ndarray] = {}

    # ── Startup: cache ou disco ─────────────────────────────────────
    def startup(self) -> None:
        """Tenta carregar do cache; se não existir, varre a pasta e salva."""
        if self._load_cache():
            return
        self.refresh()

    # ── Refresh forçado ─────────────────────────────────────────────
    def refresh(self) -> int:
        """Relê a pasta `database/`, gera embeddings e salva cache.

        Retorna a quantidade de rostos carregados.
        """
        self.known_faces.clear()
        self._scan_database()
        self._save_cache()
        return len(self.known_faces)

    # ── Verificação ─────────────────────────────────────────────────
    def verify_user(self, file_bytes: bytes) -> dict:
        """Recebe bytes de imagem e retorna resultado padronizado.

        Retorno:
            {"match": bool, "name": str, "confidence": float, "message": str}
        """
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return self._result(False, "Erro", 0.0, "Imagem inválida ou corrompida.")

        faces = self.app.get(img)
        if not faces:
            return self._result(False, "Erro", 0.0, "Nenhum rosto detectado na imagem.")

        if not self.known_faces:
            return self._result(False, "Erro", 0.0, "Nenhum rosto cadastrado no sistema.")

        query_emb = faces[0].embedding
        best_name, best_score = self._find_best_match(query_emb)

        # Faixa de confiança
        if best_score >= THRESHOLD_MATCH:
            return self._result(True, best_name, best_score, "Acesso autorizado.")

        if best_score >= THRESHOLD_DOUBT:
            return self._result(
                False, best_name, best_score,
                "Confiança insuficiente. Tente novamente com melhor iluminação.",
            )

        return self._result(False, "Desconhecido", best_score, "Rosto não reconhecido.")

    # ── Helpers públicos ────────────────────────────────────────────
    @property
    def registered_names(self) -> list[str]:
        return sorted(self.known_faces.keys())

    # ================================================================
    #  MÉTODOS INTERNOS
    # ================================================================

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (norm(a) * norm(b)))

    @staticmethod
    def _result(match: bool, name: str, confidence: float, message: str) -> dict:
        return {
            "match": match,
            "name": name,
            "confidence": round(confidence, 4),
            "message": message,
        }

    def _find_best_match(self, query_emb: np.ndarray) -> Tuple[str, float]:
        best_name = "Desconhecido"
        best_score = 0.0
        for name, known_emb in self.known_faces.items():
            score = self._cosine_similarity(query_emb, known_emb)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name, best_score

    # ── Scan da pasta database/ ─────────────────────────────────────
    def _scan_database(self, folder: Path = DATABASE_DIR) -> None:
        if not folder.exists():
            print(f"[FaceService] Pasta '{folder}' não encontrada.")
            return

        print(f"[FaceService] Varrendo pasta '{folder}'…")
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() not in _IMAGE_EXTS:
                continue

            try:
                img = cv2.imread(str(file))
                if img is None:
                    print(f"  ⚠ Não foi possível ler: {file.name}")
                    continue

                faces = self.app.get(img)
                if not faces:
                    print(f"  ⚠ Nenhum rosto detectado em: {file.name}")
                    continue

                name = file.stem
                self.known_faces[name] = faces[0].embedding
                print(f"  ✔ Cadastrado: {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ✖ Erro ao processar '{file.name}': {exc}")

        print(f"[FaceService] {len(self.known_faces)} rosto(s) carregado(s).\n")

    # ── Pickle: salvar ──────────────────────────────────────────────
    def _save_cache(self) -> None:
        try:
            data = {name: emb.tolist() for name, emb in self.known_faces.items()}
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[FaceService] Cache salvo em '{CACHE_FILE}' ({len(data)} rosto(s)).")
        except Exception as exc:  # noqa: BLE001
            print(f"[FaceService] Falha ao salvar cache: {exc}")

    # ── Pickle: carregar ────────────────────────────────────────────
    def _load_cache(self) -> bool:
        """Retorna True se conseguiu carregar do cache com sucesso."""
        if not CACHE_FILE.exists():
            print("[FaceService] Cache não encontrado. Será gerado agora.")
            return False

        try:
            with open(CACHE_FILE, "rb") as f:
                data = pickle.load(f)  # noqa: S301

            if not isinstance(data, dict) or not data:
                print("[FaceService] Cache vazio ou formato inesperado. Regenerando…")
                return False

            self.known_faces = {
                name: np.array(emb, dtype=np.float32)
                for name, emb in data.items()
            }
            print(f"[FaceService] Cache carregado: {len(self.known_faces)} rosto(s) (load instantâneo).")
            return True

        except (pickle.UnpicklingError, EOFError, ValueError) as exc:
            print(f"[FaceService] Cache corrompido ({exc}). Regenerando…")
            return False
        except Exception as exc:  # noqa: BLE001
            print(f"[FaceService] Erro ao ler cache: {exc}. Regenerando…")
            return False
