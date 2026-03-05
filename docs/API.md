# API — Autorização Selfie (v0.2)

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
  "confidence": 0.8732,
  "message": "Acesso autorizado."
}
```

| Campo      | Tipo    | Descrição                                    |
| ---------- | ------- | -------------------------------------------- |
| match      | boolean | `true` se confiança ≥ 0.65                   |
| name       | string  | Nome da pessoa ou "Desconhecido" / "Erro"    |
| confidence | float   | Score de similaridade de cosseno (0.0 – 1.0) |
| message    | string  | Mensagem descritiva do resultado             |

**Faixas de confiança**

| Faixa        | Score       | `match` | Cor         |
| ------------ | ----------- | ------- | ----------- |
| Autorizado   | `>= 0.65`   | `true`  | Verde ✅    |
| Dúvida       | `0.50–0.64` | `false` | Amarelo ⚠️  |
| Desconhecido | `< 0.50`    | `false` | Vermelho ❌ |

**Cenários de resposta**

| Situação                    | `match` | `name`           | `confidence` | `message`                               |
| --------------------------- | ------- | ---------------- | ------------ | --------------------------------------- |
| Rosto reconhecido (>0.65)   | `true`  | `"guilherme"`    | `0.87`       | `"Acesso autorizado."`                  |
| Confiança baixa (0.50–0.64) | `false` | `"guilherme"`    | `0.58`       | `"Confiança insuficiente. Tente…"`      |
| Rosto não reconhecido       | `false` | `"Desconhecido"` | `0.31`       | `"Rosto não reconhecido."`              |
| Nenhum rosto detectado      | `false` | `"Erro"`         | `0.0`        | `"Nenhum rosto detectado na imagem."`   |
| Imagem inválida             | `false` | `"Erro"`         | `0.0`        | `"Imagem inválida ou corrompida."`      |
| Sem cadastros               | `false` | `"Erro"`         | `0.0`        | `"Nenhum rosto cadastrado no sistema."` |

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
// { match: true, name: "guilherme", confidence: 0.87, message: "Acesso autorizado." }
```

---

### `GET /refresh-db`

Força a releitura da pasta `database/`, regenera embeddings e recria o cache pickle.
Útil ao cadastrar novas fotos sem reiniciar o servidor.

**Response `200 OK`**

```json
{
  "status": "ok",
  "message": "Base atualizada com 3 rosto(s).",
  "users": ["ana", "carlos", "guilherme"]
}
```

**Exemplo cURL**

```bash
curl http://localhost:8000/refresh-db
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

## Notas de Segurança (MVP)

- CORS configurado com `allow_origins=["*"]` — qualquer origem permitida.
- Sem autenticação. Para produção, adicionar API Key ou JWT.
- Sem HTTPS. Para acesso seguro à câmera em produção, usar certificado TLS
  (o `getUserMedia` exige HTTPS fora de `localhost`).

---

## Integração Angular

Ver arquivo `angular_snippet.ts` na raiz do projeto com:

- `FaceVerifyService` — service com `verify()`, `getUsers()`, `refreshDatabase()`.
- `SelfieComponent` — componente com câmera, captura e exibição de resultado.
- Notas de configuração (imports, `apiUrl`, HTTPS).
