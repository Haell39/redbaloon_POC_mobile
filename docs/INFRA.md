# INFRA — Autorização Selfie

## Visão Geral

POC de reconhecimento facial local, sem nuvem, sem Docker.  
O backend roda no PC e o frontend é acessado pelo celular via rede local.

---

## Stack

| Camada     | Tecnologia                            | Versão mínima |
| ---------- | ------------------------------------- | ------------- |
| Linguagem  | Python                                | 3.11          |
| API        | FastAPI + Uvicorn                     | —             |
| Face AI    | InsightFace (`buffalo_l`)             | —             |
| Inferência | ONNX Runtime (`CPUExecutionProvider`) | —             |
| Visão      | OpenCV (`cv2`)                        | —             |
| Álgebra    | NumPy                                 | —             |
| Frontend   | HTML + CSS + JS vanilla               | —             |

---

## Estrutura de Pastas

```
redbaloon_POC_mobile/
├── main.py              # Entrypoint FastAPI
├── face_logic.py        # Serviço de reconhecimento facial
├── requirements.txt     # Dependências pip
│
├── database/            # Fotos de cadastro (ex: guilherme.jpg)
│   └── *.jpg / *.png / *.bmp / *.webp
│
├── static/
│   └── index.html       # Frontend mobile-first
│
├── docs/                # Esta pasta
│   ├── INFRA.md
│   ├── STATUS.md
│   └── API.md
│
└── .venv/               # Ambiente virtual Python
```

---

## Dependências

```
fastapi
uvicorn
python-multipart
numpy
opencv-python
insightface
onnxruntime
```

Instalar:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # PowerShell
pip install -r requirements.txt
```

> O InsightFace faz download automático dos pesos do modelo `buffalo_l` (~500 MB)
> no primeiro uso. Os arquivos ficam em `~/.insightface/models/buffalo_l/`.

---

## Como Rodar

```bash
# 1. Ativar o ambiente virtual
.venv\Scripts\Activate.ps1

# 2. Adicionar fotos em database/ (nome do arquivo = nome cadastrado)
#    Ex: database/guilherme.jpg

# 3. Subir a API
python main.py
```

O servidor sobe em `http://0.0.0.0:8000`.

- **No PC:** `http://localhost:8000`
- **No celular (mesma rede Wi-Fi):** `http://<IP_DO_PC>:8000`

Para descobrir o IP do PC:

```powershell
ipconfig | Select-String "IPv4"
```

---

## Modelo de Reconhecimento

| Parâmetro          | Valor                     |
| ------------------ | ------------------------- |
| Modelo             | `buffalo_l`               |
| Detector           | SCRFD (face detection)    |
| Reconhecedor       | ArcFace (embedding 512-d) |
| Similaridade       | Cosine Similarity         |
| Threshold de match | `>= 0.50`                 |
| Execução           | CPU (ONNX Runtime)        |
| Resolução detecção | 640 × 640                 |

---

## Arquitetura Simplificada

```
Celular (browser)
      │
      │  POST /verify  (multipart/form-data)
      │  GET  /users
      ▼
FastAPI (0.0.0.0:8000)
      │
      ├── FaceService.verify_user()
      │         │
      │         ├── InsightFace → embedding (512-d)
      │         └── Cosine Similarity vs. known_faces{}
      │
      └── Static Files → static/index.html
```
