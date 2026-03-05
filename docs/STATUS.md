# STATUS — Autorização Selfie

> Última atualização: Março 2026

---

## Estado Atual: ✅ POC Funcional

O sistema está operacional para demonstração local. Todas as funcionalidades
core da POC foram implementadas e testadas.

---

## O que está pronto

| Feature                                          | Status |
| ------------------------------------------------ | ------ |
| Carregamento automático da pasta `database/`     | ✅     |
| Geração de embeddings com InsightFace            | ✅     |
| API `POST /verify` (verificação facial)          | ✅     |
| API `GET /users` (listagem de cadastros)         | ✅     |
| Tratamento de erros (imagem inválida, sem rosto) | ✅     |
| Frontend mobile-first com câmera frontal         | ✅     |
| Feedback visual (verde/vermelho + confiança)     | ✅     |
| CORS liberado para acesso pelo celular           | ✅     |
| Inicialização do modelo no startup da API        | ✅     |

---

## Limitações Conhecidas (POC)

| Limitação                        | Impacto | Observação                                         |
| -------------------------------- | ------- | -------------------------------------------------- |
| Embeddings apenas em memória RAM | Médio   | Perdem-se ao reiniciar o servidor                  |
| 1 foto por pessoa no cadastro    | Médio   | Múltiplos ângulos aumentariam a precisão           |
| Sem autenticação na API          | Alto    | Qualquer dispositivo na rede pode chamar `/verify` |
| CORS `allow_origins=["*"]`       | Alto    | Adequado apenas para POC local                     |
| Execução somente em CPU          | Baixo   | Latência ~1–3 s por verificação (CPU)              |
| Sem persistência de logs         | Baixo   | Logs apenas no console                             |
| Sem rate limiting                | Baixo   | Sem proteção contra spam de requests               |

---

## Próximos Passos (se evoluir para produção)

- [ ] Persistência de embeddings em banco (SQLite / Redis / Postgres + pgvector)
- [ ] Suporte a múltiplas fotos por pessoa (média dos embeddings)
- [ ] Autenticação JWT ou API Key na rota `/verify`
- [ ] Liveness detection (anti-spoofing) para evitar foto de foto
- [ ] GPU support (`CUDAExecutionProvider`) para latência < 200 ms
- [ ] Upload de novas fotos via API (rota `POST /register`)
- [ ] Logs estruturados (ex: `structlog` ou `loguru`)
- [ ] Containerizar com Docker para portabilidade
- [ ] CI/CD + testes automatizados

---

## Métricas de Referência (ambiente de teste)

| Operação                         | Tempo estimado (CPU) |
| -------------------------------- | -------------------- |
| Startup + carregamento do modelo | ~8–15 s              |
| `load_from_disk` (1 foto)        | ~0.5–1 s             |
| `verify_user` por chamada        | ~1–3 s               |

> Tempos medidos em CPU comum (i5/i7 geração recente). Variam com hardware.

---

## Threshold de Confiança

| Score (Cosine Similarity) | Interpretação      |
| ------------------------- | ------------------ |
| `>= 0.80`                 | Match forte ✅     |
| `0.60 – 0.79`             | Match razoável ⚠️  |
| `0.50 – 0.59`             | Match limítrofe ⚠️ |
| `< 0.50`                  | Desconhecido ❌    |

---

## Ambiente Testado

- **OS:** Windows 11
- **Python:** 3.11
- **Modelo:** InsightFace `buffalo_l`
- **Hardware:** CPU (sem GPU)
- **Frontend:** Chrome Mobile (Android) via Wi-Fi local
