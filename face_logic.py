"""
face_logic.py — Serviço de reconhecimento facial (InsightFace / buffalo_l).

Responsabilidades:
  • Carregar imagens da pasta `database/` e gerar embeddings.
  • Comparar um rosto recebido (bytes) com os embeddings conhecidos
    usando Similaridade de Cosseno.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# Extensões de imagem aceitas
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATABASE_DIR = Path(__file__).resolve().parent / "database"


class FaceService:
    """Serviço singleton de reconhecimento facial."""

    def __init__(self) -> None:
        print("[FaceService] Inicializando modelo InsightFace (buffalo_l / CPU)…")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Cadastro automático
    # ------------------------------------------------------------------
    def load_from_disk(self, folder: Path = DATABASE_DIR) -> None:
        """Percorre *folder* e registra embeddings de cada rosto encontrado."""
        if not folder.exists():
            print(f"[FaceService] Pasta '{folder}' não encontrada. Nenhum rosto carregado.")
            return

        print(f"[FaceService] Lendo pasta '{folder}'…")
        for file in sorted(folder.iterdir()):
            if file.suffix.lower() not in _IMAGE_EXTS:
                continue  # pula arquivos não-imagem

            try:
                img = cv2.imread(str(file))
                if img is None:
                    print(f"  ⚠ Não foi possível ler: {file.name}")
                    continue

                faces = self.app.get(img)
                if not faces:
                    print(f"  ⚠ Nenhum rosto detectado em: {file.name}")
                    continue

                # Usa o primeiro rosto encontrado
                embedding = faces[0].embedding
                name = file.stem  # nome sem extensão
                self.known_faces[name] = embedding
                print(f"  ✔ Cadastrado: {name}")
            except Exception as exc:  # noqa: BLE001
                print(f"  ✖ Erro ao processar '{file.name}': {exc}")

        total = len(self.known_faces)
        print(f"[FaceService] {total} rosto(s) carregado(s) com sucesso.\n")

    # ------------------------------------------------------------------
    # Verificação
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Retorna a similaridade de cosseno entre dois vetores."""
        return float(np.dot(a, b) / (norm(a) * norm(b)))

    def verify_user(self, file_bytes: bytes) -> Tuple[str, float]:
        """
        Recebe os bytes de uma imagem, detecta o rosto, gera embedding e
        compara com todos os rostos conhecidos.

        Retorna:
            (nome, confiança)  — se confiança < 0.5 ➜ ("Desconhecido", confiança)
        """
        # Decodifica bytes → imagem OpenCV (BGR)
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return ("Erro: imagem inválida", 0.0)

        faces = self.app.get(img)
        if not faces:
            return ("Erro: nenhum rosto detectado", 0.0)

        query_emb = faces[0].embedding

        best_name = "Desconhecido"
        best_score = 0.0

        for name, known_emb in self.known_faces.items():
            score = self._cosine_similarity(query_emb, known_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < 0.5:
            return ("Desconhecido", round(best_score, 4))

        return (best_name, round(best_score, 4))
