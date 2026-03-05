# Identity Service — Reconhecimento Facial

Microserviço de verificação facial (InsightFace / ArcFace) exposto via FastAPI.  
Projetado para integração com backends Node.js e frontends Angular.

---

## Arquitetura

```
┌─────────────────────────────────────────────┐
│  Angular App (:4200)  /  Node.js Backend    │
│                                             │
│  POST /verify   (multipart/form-data)       │
│  GET  /refresh-db                           │
│  GET  /users                                │
│  GET  /health                               │
└────────────────────┬────────────────────────┘
                     │  HTTP
                     ▼
┌────────────────────────────────────────────┐
│  Identity Service (:8000)                  │
│                                            │
│  FastAPI                                   │
│    └── FaceService                         │
│          ├── InsightFace (buffalo_l / CPU)  │
│          ├── Cosine Similarity             │
│          └── Cache: data/face_encodings.pkl│
└────────────────────────────────────────────┘
```

---

## Estrutura do Projeto

```
├── src/
│   ├── main.py                 # Entrypoint FastAPI
│   ├── config.py               # Configurações centralizadas
│   └── services/
│       └── face_service.py     # Lógica de reconhecimento facial
├── data/
│   └── face_encodings.pkl      # Cache de embeddings (auto-gerado)
├── database/
│   └── *.jpg / *.png           # Fotos de cadastro (nome = ID)
├── logs/
│   └── service.log             # Log de execução
├── static/
│   └── index.html              # Test client (browser)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Início Rápido

### Rodando Local (sem Docker)

```bash
# 1. Criar e ativar virtualenv
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # Linux/Mac

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Adicionar fotos na pasta database/
#    Nome do arquivo = ID do funcionário (ex: guilherme.jpg)

# 4. Iniciar o serviço
python -m src.main
```

O serviço sobe em `http://0.0.0.0:8000`.

### Rodando com Docker

```bash
docker compose up --build
```

Os volumes `data/`, `database/` e `logs/` são mapeados. Atualize fotos em `database/` e chame `/refresh-db` sem rebuild.

---

## API Reference

### `POST /verify`

Envia uma selfie e recebe a verificação facial.

**Request:**

```
Content-Type: multipart/form-data
```

| Campo | Tipo       | Obrigatório | Descrição         |
| ----- | ---------- | ----------- | ----------------- |
| file  | UploadFile | Sim         | Imagem (JPEG/PNG) |

**Response `200`:**

```json
{
  "id": "guilherme",
  "status": "match",
  "confidence": 0.8732,
  "message": "Acesso autorizado."
}
```

| Campo      | Tipo   | Valores possíveis                                   |
| ---------- | ------ | --------------------------------------------------- |
| id         | string | Nome do funcionário ou `"unknown"`                  |
| status     | string | `"match"` \| `"no_match"` \| `"doubt"` \| `"error"` |
| confidence | float  | 0.0 – 1.0                                           |
| message    | string | Descrição legível do resultado                      |

**Faixas de confiança:**

| Score         | Status     | Significado                  |
| ------------- | ---------- | ---------------------------- |
| `>= 0.65`     | `match`    | Identidade confirmada        |
| `0.50 – 0.64` | `doubt`    | Tente novamente (iluminação) |
| `< 0.50`      | `no_match` | Rosto não reconhecido        |

---

### `GET /refresh-db`

Força releitura da pasta `database/` e recria o cache `.pkl`.  
Use após adicionar/remover fotos.

```json
{
  "status": "ok",
  "message": "Base atualizada com 3 rosto(s).",
  "users": ["ana", "carlos", "guilherme"]
}
```

---

### `GET /users`

Lista funcionários cadastrados (debug).

```json
{
  "total": 3,
  "users": ["ana", "carlos", "guilherme"]
}
```

---

### `GET /health`

Healthcheck para Docker / balanceadores.

```json
{
  "status": "healthy",
  "model": "buffalo_l",
  "registered_faces": 3
}
```

---

## Exemplos de Integração (Node.js)

### cURL

```bash
# Verificar rosto
curl -X POST http://localhost:8000/verify \
     -F "file=@selfie.jpg"

# Atualizar base
curl http://localhost:8000/refresh-db

# Listar cadastrados
curl http://localhost:8000/users

# Health check
curl http://localhost:8000/health
```

### Node.js (axios + form-data)

```javascript
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

async function verifyFace(imagePath) {
  const form = new FormData();
  form.append("file", fs.createReadStream(imagePath));

  const { data } = await axios.post("http://localhost:8000/verify", form, {
    headers: form.getHeaders(),
  });

  console.log(data);
  // { id: "guilherme", status: "match", confidence: 0.87, message: "Acesso autorizado." }
  return data;
}
```

### Angular (HttpClient)

```typescript
verify(imageBlob: Blob): Observable<VerifyResponse> {
  const formData = new FormData();
  formData.append('file', imageBlob, 'selfie.jpg');
  return this.http.post<VerifyResponse>('http://localhost:8000/verify', formData);
}
```

---

## Modelo de IA

| Parâmetro    | Valor                     |
| ------------ | ------------------------- |
| Framework    | InsightFace               |
| Modelo       | `buffalo_l`               |
| Detector     | SCRFD                     |
| Reconhecedor | ArcFace (embedding 512-d) |
| Similaridade | Cosine Similarity         |
| Execução     | CPU (ONNX Runtime)        |
| Resolução    | 640 × 640                 |

---

## Variáveis de Ambiente

Não há variáveis obrigatórias. Tudo é configurado via `src/config.py`.  
Para alterar porta ou thresholds, edite o arquivo diretamente ou adapte para `os.environ`.
