# STATUS — Autorização Selfie

> Última atualização: Março 2026

---

## Estado Atual: ✅ MVP Funcional (v0.2)

O sistema evoluiu de POC para MVP. Inclui persistência de embeddings
via pickle, três faixas de confiança e contrato de API padronizado
para integração com frontend Angular.

---

## O que está pronto

| Feature                                            | Status |
| -------------------------------------------------- | ------ |
| Carregamento automático da pasta `database/`       | ✅     |
| Geração de embeddings com InsightFace              | ✅     |
| **Cache pickle** (`face_encodings.pkl`)            | ✅ NEW |
| **Load instantâneo** quando cache existe           | ✅ NEW |
| API `POST /verify` (contrato padronizado)          | ✅ UPD |
| API `GET /users` (listagem de cadastros)           | ✅     |
| **API `GET /refresh-db`** (recarrega sem restart)  | ✅ NEW |
| **3 faixas de confiança** (verde/amarelo/vermelho) | ✅ NEW |
| **Campo `message`** no retorno JSON                | ✅ NEW |
| Tratamento de erros (imagem inválida, sem rosto)   | ✅     |
| Test client mobile-first com câmera frontal        | ✅ UPD |
| **Status bar** (API online, qtd cadastros)         | ✅ NEW |
| **Botão refresh DB** no test client                | ✅ NEW |
| **Snippet Angular** (Service + Component)          | ✅ NEW |
| CORS liberado para acesso pelo celular             | ✅     |
| Inicialização do modelo no startup da API          | ✅     |

---

## Limitações Conhecidas (POC)

| Limitação                            | Impacto   | Observação                                         |
| ------------------------------------ | --------- | -------------------------------------------------- |
| ~~Embeddings apenas em memória RAM~~ | ~~Médio~~ | **RESOLVIDO** — cache pickle em disco              |
| 1 foto por pessoa no cadastro        | Médio     | Múltiplos ângulos aumentariam a precisão           |
| Sem autenticação na API              | Alto      | Qualquer dispositivo na rede pode chamar `/verify` |
| CORS `allow_origins=["*"]`           | Alto      | Adequado apenas para POC local                     |
| Execução somente em CPU              | Baixo     | Latência ~1–3 s por verificação (CPU)              |
| Sem persistência de logs             | Baixo     | Logs apenas no console                             |
| Sem rate limiting                    | Baixo     | Sem proteção contra spam de requests               |

---

## Próximos Passos (se evoluir para produção)

- [x] ~~Persistência de embeddings~~ → Implementado via pickle (`face_encodings.pkl`)
- [x] ~~Refresh de base sem restart~~ → Endpoint `GET /refresh-db`
- [x] ~~Contrato JSON padronizado~~ → `{ match, name, confidence, message }`
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

## Threshold de Confiança (v0.2)

| Score (Cosine Similarity) | Faixa           | `match` | Cor no frontend |
| ------------------------- | --------------- | ------- | --------------- |
| `>= 0.65`                 | Autorizado ✅   | `true`  | Verde           |
| `0.50 – 0.64`             | Dúvida ⚠️       | `false` | Amarelo         |
| `< 0.50`                  | Desconhecido ❌ | `false` | Vermelho        |

---

## Ambiente Testado

- **OS:** Windows 11
- **Python:** 3.11
- **Modelo:** InsightFace `buffalo_l`
- **Hardware:** CPU (sem GPU)
- **Frontend:** Chrome Mobile (Android) via Wi-Fi local
