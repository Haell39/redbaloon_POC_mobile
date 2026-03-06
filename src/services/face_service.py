"""
face_service.py — Serviço de reconhecimento facial (Dual-Bank + CSV Enrichment).

Responsabilidades:
  • Carregar embeddings pré-computados de PKL(s) na pasta database/ (responsáveis)
    e database_equip/ (equipe interna).
  • Indexar tabela CSV de responsáveis para enriquecimento dos matches.
  • Comparar rosto recebido (bytes) via Cosine Similarity contra ambos os bancos.
  • Retornar contrato padronizado com dados enriquecidos do CSV quando disponíveis.
"""

from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

from src.config import (
    CSV_FILE,
    DATABASE_DIR,
    DATABASE_EQUIP_DIR,
    DET_SIZE,
    MODEL_NAME,
    MODEL_PROVIDERS,
    REGISTER_PKL_FILE,
    THRESHOLD_DOUBT,
    THRESHOLD_MATCH,
)

logger = logging.getLogger("identity-service")

# Tipo interno de cada entrada de embedding carregada de um PKL
FaceEntry = Dict  # {id, filename, embedding: np.ndarray}


class FaceService:
    """Serviço de reconhecimento facial com suporte a dois bancos (PKL) e CSV."""

    def __init__(self) -> None:
        logger.info("Inicializando modelo InsightFace (%s / CPU)…", MODEL_NAME)
        self.app = FaceAnalysis(name=MODEL_NAME, providers=MODEL_PROVIDERS)
        self.app.prepare(ctx_id=0, det_size=DET_SIZE)

        # Banco de responsáveis: filename → FaceEntry
        self.known_resp: Dict[str, FaceEntry] = {}
        # Banco de equipe: filename → FaceEntry
        self.known_equip: Dict[str, FaceEntry] = {}
        # Índice CSV: foto (filename) → row dict
        self.csv_index: Dict[str, dict] = {}

    # ── Startup ─────────────────────────────────────────────────────
    def startup(self) -> None:
        """Carrega os dois bancos de PKL e o índice do CSV."""
        resp_count  = self._load_bank(DATABASE_DIR,       self.known_resp,  "responsáveis")
        equip_count = self._load_bank(DATABASE_EQUIP_DIR, self.known_equip, "equipe")
        self._load_csv()
        logger.info(
            "Pronto — %d responsável(is), %d membro(s) de equipe, %d entrada(s) no CSV.",
            resp_count, equip_count, len(self.csv_index),
        )

    # ── Refresh forçado ─────────────────────────────────────────────
    def refresh(self) -> dict:
        """Relê PKLs e CSV. Retorna contagem de cada banco."""
        self.known_resp.clear()
        self.known_equip.clear()
        self.csv_index.clear()
        resp  = self._load_bank(DATABASE_DIR,       self.known_resp,  "responsáveis")
        equip = self._load_bank(DATABASE_EQUIP_DIR, self.known_equip, "equipe")
        self._load_csv()
        return {"resp": resp, "equip": equip, "csv": len(self.csv_index)}

    # ── Registro de novo rosto ──────────────────────────────────────
    def register(self, resp_id: str, file_bytes: bytes) -> dict:
        """Vetoriza uma foto e cadastra o rosto no banco de responsáveis.

        Args:
            resp_id:    ID do responsável (vem do backend Node)
            file_bytes: bytes da imagem (JPEG/PNG)

        Retorno:
            {"status": "ok"|"error", "id": str, "message": str}
        """
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("[register] Imagem inválida para id=%s.", resp_id)
            return {"status": "error", "id": resp_id, "message": "Imagem inválida ou corrompida."}

        faces = self.app.get(img)
        if not faces:
            logger.warning("[register] Nenhum rosto detectado para id=%s.", resp_id)
            return {"status": "error", "id": resp_id, "message": "Nenhum rosto detectado na imagem."}

        if len(faces) > 1:
            logger.warning("[register] Múltiplos rostos detectados para id=%s. Usando o primeiro.", resp_id)

        embedding = faces[0].embedding
        filename = f"{resp_id}.jpg"

        self.known_resp[filename] = {
            "id": str(resp_id),
            "filename": filename,
            "embedding": embedding,
        }

        self._persist_registered()

        logger.info("[register] Rosto cadastrado: id=%s (total resp=%d).", resp_id, self.total_resp)
        return {
            "status": "ok",
            "id": resp_id,
            "message": f"Rosto cadastrado com sucesso. Total: {self.total_resp} responsável(is).",
        }

    # ── Remoção de rosto ────────────────────────────────────────────
    def unregister(self, resp_id: str) -> dict:
        """Remove um rosto do banco de responsáveis."""
        filename = f"{resp_id}.jpg"
        if filename not in self.known_resp:
            return {"status": "error", "id": resp_id, "message": "ID não encontrado na base."}

        del self.known_resp[filename]
        self._persist_registered()
        logger.info("[unregister] Rosto removido: id=%s.", resp_id)
        return {"status": "ok", "id": resp_id, "message": "Rosto removido com sucesso."}

    # ── Verificação ─────────────────────────────────────────────────
    def verify(self, file_bytes: bytes) -> dict:
        """Recebe bytes de imagem e retorna resultado padronizado.

        Contrato de retorno:
            {
                "id":         str,    # id do PKL ou "unknown"
                "filename":   str,    # filename do PKL ou ""
                "status":     str,    # "match" | "doubt" | "no_match" | "error"
                "confidence": float,  # 0.0 – 1.0
                "source":     str,    # "resp" | "equip" | "unknown"
                "message":    str,
                # campos abaixo presentes somente quando source == "resp" e há dados no CSV
                "nome":       str | None,
                "cpf":        str | None,
                "numero":     str | None,
                "ativo":      str | None,
                "origem":     str | None,
            }
        """
        total = len(self.known_resp) + len(self.known_equip)
        if total == 0:
            return self._err("Nenhum rosto cadastrado no sistema.")

        # Decodifica imagem
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("Imagem recebida inválida/corrompida.")
            return self._err("Imagem inválida ou corrompida.")

        # Detecção
        faces = self.app.get(img)
        if not faces:
            logger.warning("Nenhum rosto detectado.")
            return self._err("Nenhum rosto detectado na imagem.")

        query_emb = faces[0].embedding

        # Busca nos dois bancos — escolhe o melhor score geral
        best_entry: Optional[FaceEntry] = None
        best_score: float = 0.0
        best_source: str = "unknown"

        for source, bank in (("resp", self.known_resp), ("equip", self.known_equip)):
            if not bank:
                continue
            entry, score = self._find_best_match(query_emb, bank)
            if score > best_score:
                best_score  = score
                best_entry  = entry
                best_source = source

        if best_entry is None or best_score < THRESHOLD_DOUBT:
            logger.info("No match (%.4f)", best_score)
            return self._no_match(best_score)

        filename   = best_entry.get("filename", "")
        pkl_id     = str(best_entry.get("id", ""))
        csv_row    = self.csv_index.get(filename)  # None se equipe ou sem correspondência no CSV
        csv_id     = csv_row["id"] if csv_row else pkl_id
        display_id = csv_id or pkl_id

        if best_score >= THRESHOLD_MATCH:
            logger.info("Match: %s [%s] (%.4f)", filename, best_source, best_score)
            return self._build_response(
                display_id, filename, "match", best_score, best_source,
                "Acesso autorizado.", csv_row,
            )

        # Faixa de dúvida
        logger.info("Dúvida: %s [%s] (%.4f)", filename, best_source, best_score)
        return self._build_response(
            display_id, filename, "doubt", best_score, best_source,
            "Confiança insuficiente. Tente novamente com melhor iluminação.", csv_row,
        )

    # ── Propriedades públicas ───────────────────────────────────────
    @property
    def registered_names_resp(self) -> List[str]:
        return sorted(self.known_resp.keys())

    @property
    def registered_names_equip(self) -> List[str]:
        return sorted(self.known_equip.keys())

    @property
    def total_resp(self) -> int:
        return len(self.known_resp)

    @property
    def total_equip(self) -> int:
        return len(self.known_equip)

    # ================================================================
    #  INTERNALS
    # ================================================================

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = norm(a), norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _find_best_match(
        self, query_emb: np.ndarray, bank: Dict[str, FaceEntry]
    ) -> Tuple[Optional[FaceEntry], float]:
        best_entry: Optional[FaceEntry] = None
        best_score = 0.0
        for entry in bank.values():
            emb   = entry["embedding"]
            score = self._cosine_similarity(query_emb, emb)
            if score > best_score:
                best_score = score
                best_entry = entry
        return best_entry, best_score

    @staticmethod
    def _build_response(
        id_: str, filename: str, status: str, confidence: float,
        source: str, message: str, csv_row: Optional[dict],
    ) -> dict:
        resp: dict = {
            "id":         id_,
            "status":     status,
            "confidence": round(confidence, 4),
            "source":     source,
            "message":    message,
        }
        if csv_row:
            for key in ("nome", "cpf", "numero", "ativo", "origem"):
                val = csv_row.get(key)
                if val:  # Só adiciona se não for vazio ou null
                    resp[key] = val
        return resp

    @staticmethod
    def _err(message: str) -> dict:
        return {
            "id":         "unknown",
            "status":     "error",
            "confidence": 0.0,
            "source":     "unknown",
            "message":    message,
        }

    @staticmethod
    def _no_match(confidence: float) -> dict:
        return {
            "id":         "unknown",
            "status":     "no_match",
            "confidence": round(confidence, 4),
            "source":     "unknown",
            "message":    "Rosto não reconhecido.",
        }

    # ── Carga de banco PKL ──────────────────────────────────────────
    def _load_bank(
        self,
        folder: Path,
        target: Dict[str, FaceEntry],
        label: str,
    ) -> int:
        """Carrega todos os PKLs de uma pasta no dict target.

        Suporta:
          - Lista de dicts: [{id, filename, embedding}, ...]  (formato Vetorizator)
          - Dict: {name → embedding_list}                     (formato legado)
        """
        if not folder.exists():
            logger.warning("Pasta '%s' não encontrada (%s).", folder, label)
            return 0

        pkl_files = sorted(folder.glob("*.pkl"))
        if not pkl_files:
            logger.warning("Nenhum .pkl encontrado em '%s' (%s).", folder, label)
            return 0

        loaded = 0
        for pkl_path in pkl_files:
            try:
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)  # noqa: S301

                # Formato Vetorizator: list of dicts
                if isinstance(data, list):
                    for item in data:
                        filename = item.get("filename", "")
                        emb_raw  = item.get("embedding")
                        if not filename or emb_raw is None:
                            continue
                        emb = np.array(emb_raw, dtype=np.float32)
                        target[filename] = {
                            "id":        str(item.get("id", Path(filename).stem)),
                            "filename":  filename,
                            "embedding": emb,
                        }
                        loaded += 1

                # Formato legado: dict {name → embedding}
                elif isinstance(data, dict):
                    for name, emb_raw in data.items():
                        emb = np.array(emb_raw, dtype=np.float32)
                        # Usa nome sem extensão como filename para compatibilidade
                        filename = name if "." in name else name
                        target[filename] = {
                            "id":        Path(name).stem,
                            "filename":  filename,
                            "embedding": emb,
                        }
                        loaded += 1

                else:
                    logger.warning("Formato inesperado em '%s'. Ignorando.", pkl_path.name)

                logger.info("  [%s] %s — %d rosto(s) carregado(s).", label, pkl_path.name, loaded)

            except (pickle.UnpicklingError, EOFError, ValueError) as exc:
                logger.warning("PKL corrompido '%s': %s", pkl_path.name, exc)
            except Exception:
                logger.exception("Erro ao carregar '%s'.", pkl_path.name)

        logger.info("[%s] Total: %d rosto(s).", label, len(target))
        return len(target)

    # ── Carga do CSV ────────────────────────────────────────────────
    def _load_csv(self) -> None:
        """Indexa o CSV de responsáveis por foto (filename)."""
        if not CSV_FILE.exists():
            logger.warning("CSV não encontrado: %s", CSV_FILE)
            return
        try:
            with open(CSV_FILE, newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    foto = row.get("foto", "").strip()
                    if foto:
                        self.csv_index[foto] = row
            logger.info("CSV indexado: %d entrada(s) [chave=foto].", len(self.csv_index))
        except Exception:
            logger.exception("Erro ao carregar CSV '%s'.", CSV_FILE)

    # ── Persistência de registros dinâmicos ──────────────────────────
    def _persist_registered(self) -> None:
        """Salva todos os rostos do banco resp em registered_faces.pkl.

        Formato: lista de dicts [{id, filename, embedding}, ...] (Vetorizator-compatible).
        """
        try:
            REGISTER_PKL_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = []
            for entry in self.known_resp.values():
                data.append({
                    "id": entry["id"],
                    "filename": entry["filename"],
                    "embedding": entry["embedding"],
                })
            with open(REGISTER_PKL_FILE, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Registros persistidos: %s (%d rosto(s)).", REGISTER_PKL_FILE.name, len(data))
        except Exception:
            logger.exception("Falha ao persistir registros.")
