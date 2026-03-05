# API — Autorização Selfie

Base URL: `http://<HOST>:8000`

---

## Endpoints

### `POST /verify`

Recebe uma imagem de selfie e retorna se o rosto bate com alguém cadastrado.

**Request**

```
Content-Type: multipart/form-data
```

| Campo | Tipo       | Obrigatório | Descrição                  |
| ----- | ---------- | ----------- | -------------------------- |
| file  | UploadFile | ✅          | Imagem do rosto (JPEG/PNG) |

**Response `200 OK`**

```json
{
  "match": true,
  "name": "guilherme",
  "confidence": 0.8732
}
```

| Campo      | Tipo    | Descrição                                                     |
| ---------- | ------- | ------------------------------------------------------------- |
| match      | boolean | `true` se confiança ≥ 0.5 e rosto reconhecido                 |
| name       | string  | Nome do arquivo cadastrado (sem extensão) ou mensagem de erro |
| confidence | float   | Score de similaridade de cosseno (0.0 – 1.0)                  |

**Cenários de resposta**

| Situação                   | `match` | `name`                           | `confidence` |
| -------------------------- | ------- | -------------------------------- | ------------ |
| Rosto reconhecido          | `true`  | `"guilherme"`                    | `0.87`       |
| Rosto não reconhecido      | `false` | `"Desconhecido"`                 | `0.31`       |
| Nenhum rosto detectado     | `false` | `"Erro: nenhum rosto detectado"` | `0.0`        |
| Arquivo de imagem inválido | `false` | `"Erro: imagem inválida"`        | `0.0`        |

**Exemplo cURL**

```bash
curl -X POST http://localhost:8000/verify \
     -F "file=@caminho/para/selfie.jpg"
```

**Exemplo JavaScript (frontend)**

```js
const form = new FormData();
form.append("file", blob, "selfie.jpg");

const response = await fetch("/verify", { method: "POST", body: form });
const data = await response.json();
// { match: true, name: "guilherme", confidence: 0.87 }
```

---

### `GET /users`

Retorna a lista de nomes carregados da pasta `database/`. Útil para debug.

**Response `200 OK`**

```json
{
  "users": ["guilherme", "ana", "carlos"]
}
```

| Campo | Tipo            | Descrição                                        |
| ----- | --------------- | ------------------------------------------------ |
| users | array of string | Nomes registrados (nome do arquivo sem extensão) |

**Exemplo cURL**

```bash
curl http://localhost:8000/users
```

---

### `GET /`

Serve o frontend mobile (`static/index.html`).

Acesse pelo navegador do celular:

```
http://<IP_DO_PC>:8000
```

---

## Códigos de Erro HTTP

| Código | Quando ocorre                                          |
| ------ | ------------------------------------------------------ |
| `200`  | Sempre (erros de negócio estão no payload JSON)        |
| `422`  | Campo `file` ausente ou tipo inválido (FastAPI padrão) |
| `500`  | Erro interno inesperado no servidor                    |

---

## Notas de Segurança (POC)

- CORS configurado com `allow_origins=["*"]` — qualquer origem permitida.
- Sem autenticação. Para produção, adicionar API Key ou JWT.
- Sem HTTPS. Para acesso seguro à câmera em produção, usar certificado TLS
  (o `getUserMedia` exige HTTPS fora de `localhost`).
