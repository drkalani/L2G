# L2G Platform

End-to-end system for **disease-agnostic** literature mining: train a **binary relevance** classifier on your labeled abstracts (any condition), run **gene/protein NER**, and **normalize** symbols (MyGene + optional Wikipedia fallback). Backend: **FastAPI**. UI: **React (Vite)**.

## Features

- **Projects**: separate workspace per disease/study (`disease_key` is metadata; models are trained on *your* labels).
- **Training**: holdout validation or **stratified k-fold**; choose **CUDA**, **Apple Metal (MPS)**, or **CPU**; tune LR, epochs, batch sizes, sequence length, FP16 (auto on CUDA by default).
- **Pipeline**: run **classify only**, **NER only**, **normalize only**, or **full** workflow.
- **Datasets**: import articles or mentions from **CSV / Excel / pickle**; export each step or a **bundle** (multi-sheet Excel or dict of DataFrames in PKL); CSV templates from the UI and API.
- **REST API**: OpenAPI at `/docs` when the server is running.

Pipeline runs write CSV snapshots under `data/projects/<id>/outputs/last_run/` (classification, mentions, normalized) so exports always have a stable on-disk source.

## Requirements

- Python **3.13.7+** (backend runtime; Bent remains isolated on Python 3.10.x)
- Node.js **18+** (for the frontend)
- PyTorch with the backend you need ([pytorch.org](https://pytorch.org))

## Setup

```bash
cd /path/to/L2G
python3.13 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[api]"
```

### Backend

```bash
cd backend
export HF_TOKEN=your_hf_token_here  # Optional: improves HF download speed and rate limits
export PYTHONPATH=.
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Or from the repo root:

```bash
export HF_TOKEN=your_hf_token_here  # Optional
chmod +x scripts/run_backend.sh
./scripts/run_backend.sh
```

### Environment file (optional)

Create `.env` from the template when you want persistent defaults for local execution:

```bash
cp .env.example .env
```

Then edit values:

```bash
HF_TOKEN=your_hf_token_here
BENT_SERVICE_URL=http://127.0.0.1:8010
BENT_SERVICE_TIMEOUT_SECONDS=30
HF_HOME=$HOME/.cache/huggingface
HF_HUB_CACHE=$HOME/.cache/huggingface/hub
HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface/hub
```

If `.env` is missing, `./scripts/run_backend.sh` now falls back to process environment and safe defaults:
- HF token: empty (`""`)
- caches: `$HOME/.cache/huggingface` and derived hub paths
- bent URL: empty (no service override)
- bent timeout: `30`

### Bent runtime setup (optional legacy method)

For local backend execution (no Docker), run:

```bash
chmod +x scripts/setup_bent_runtime.sh
./scripts/setup_bent_runtime.sh
```

By default this creates or reuses a dedicated Python 3.10.x virtual environment at `./.venv-bent` so your main project virtualenv is unchanged.

```bash
PY_BIN=python3.10 ./scripts/setup_bent_runtime.sh
BENT_VENV="$PWD/.venv-bent" ./scripts/setup_bent_runtime.sh
RUN_SETUP=1 ./scripts/setup_bent_runtime.sh
```

If `python3.10` is reported as `3.10.20`, Bent cannot be installed because `bent==0.0.80` requires `Python <= 3.10.13`.
Use a dedicated `3.10.13` interpreter for Bent setup, for example:

```bash
# Example with pyenv
pyenv install 3.10.13
PY_BIN="$(pyenv root)/versions/3.10.13/bin/python" ./scripts/setup_bent_runtime.sh
```

This script:
- verifies Python 3.10.x compatibility (`<=3.10.13`)
- installs `bent==0.0.80` (latest release compatible with Python 3.10.13) into a dedicated Bent virtualenv
- writes `frontend/.env.local` with `VITE_API_BASE=http://127.0.0.1:8000`

To run a Bent-only step manually:

```bash
source .venv-bent/bin/activate
bent_setup
deactivate
```

To expose Bent as a microservice from that venv:

```bash
chmod +x scripts/run_bent_service.sh
./scripts/run_bent_service.sh

pkill -f "bent_service.main" || true
pkill -f "run_bent_service.sh" || true
BENT_SETUP=0 PYTHONUNBUFFERED=1 BENT_SERVICE_HOST=127.0.0.1 ./scripts/run_bent_service.sh

curl -i -m 20 -X POST http://127.0.0.1:8010/annotate \
  -H 'Content-Type: application/json' \
  -d '{"pairs":[{"pmid":"test-1","text":"TP53 is a tumor suppressor gene.","text_index":0}]}'

```

Then set backend env before launch:

```bash
export BENT_SERVICE_URL=http://127.0.0.1:8010
```

For Docker compose deployments, a dedicated optional Bent runtime service is defined in
`docker-compose.yml` as `bent-runtime` and serves `POST /annotate` on port `8010`.
Run it before enabling Bent jobs:

```bash
chmod +x scripts/setup_docker_bent.sh
./scripts/setup_docker_bent.sh
```

Optional helper:
- set `RUN_SETUP=0` to skip running `bent_setup` inside the container
- set `WRITE_ENV=0` to avoid overwriting `frontend/.env.local`
- set `COMPOSE_FILE=...` to use a specific compose file

When using the container profile, set:

```bash
export BENT_SERVICE_URL=http://bent-runtime:8010
```

Optional tuning:

```bash
export BENT_SERVICE_TIMEOUT_SECONDS=60
```

To include `bent-runtime` during a single compose bootstrap:

```bash
docker compose --profile bent up --build
```

Data and trained models are stored under `data/projects/` (created automatically).

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). API calls are proxied to `http://127.0.0.1:8000` under `/api/*`.

#### Frontend API base configuration

- For local development, keep `VITE_API_BASE` unset unless you used `scripts/setup_bent_runtime.sh`,
  which writes `frontend/.env.local` explicitly.
- For Colab public endpoint, set:

```bash
cd frontend
cat > .env.local <<'EOF'
VITE_API_BASE=https://<your-ngrok-domain>
EOF
npm run dev
```

- You can also run with explicit env variable at command line:

```bash
VITE_API_BASE=https://<your-ngrok-domain> npm run dev
```

### Colab backend template

Run this notebook on Google Colab to expose the backend using a free GPU.

1. Configure the Colab launch target in frontend env:
   - Copy frontend env example if needed:
   ```bash
   cd frontend
   cp .env.example .env.local
   ```
   - Set:
   ```bash
   VITE_COLAB_GITHUB_USERNAME=drkalani
   VITE_COLAB_GITHUB_REPO=L2G
   VITE_COLAB_GITHUB_BRANCH=main
   ```
   Replace values for your fork if needed.

2. Open L2G home page and click **Open Colab notebook** (it points to
   `https://colab.research.google.com/github/$VITE_COLAB_GITHUB_USERNAME/$VITE_COLAB_GITHUB_REPO/blob/$VITE_COLAB_GITHUB_BRANCH/colab_l2g_backend.ipynb`).

3. In Colab notebook, run the environment setup cells.
   - It installs backend dependencies, starts FastAPI in background, and opens an ngrok tunnel.
   - Replace `REPO_URL` in the notebook with your own repository URL if needed.
   - For Colab Python 3.12 (or older), the notebook now skips package editable install
     and installs API/backend dependencies directly as a compatibility path.
   - If editable install is attempted on newer runtimes, it first retries with
     `--ignore-requires-python`, then falls back to explicit dependency installation.
   - If `/health` is not reachable, open `colab_uvicorn_backend.log` in the notebook working
     directory (`/content/<your-repo>`) to view the startup traceback and fix the actual import/runtime error.
   - For ngrok, set `NGROK_AUTHTOKEN` (or `NGROK_TOKEN` / `NGROK_API_TOKEN`) in Colab environment, or enter it when prompted in step 6.
   - If frontend shows `Failed to fetch`, restart from step 4 after setting:
     - `FRONTEND_ORIGIN` in Colab to your UI origin (example: `http://localhost:5173`), or
     - manually edit `CORS_ORIGINS` in notebook step 4 to include your origin.
   - If CORS is still blocked, set temporary Colab dev fallback:
     - `CORS_ALLOW_ALL=true` and (recommended) `CORS_ALLOW_CREDENTIALS=false`
     in step 4 before backend startup, then restart from step 4.
   - If the browser or DevTools show `ngrok-error-code: ERR_NGROK_6024` and `content-type: text/html` for `/health`, that is **ngrok’s free-tier interstitial page**, not your API. The L2G UI sends `ngrok-skip-browser-warning` automatically for ngrok hosts; rebuild/redeploy the frontend if you still see the HTML page.
   - If CORS still fails after that, open `colab_uvicorn_backend.log` and confirm the startup line includes:
     `Loaded CORS origins: [...]` and that your exact origin is present.

You can also run a one-command smoke check (works both local and Colab) after backend is up:

```bash
python scripts/colab_smoke_check.py --base-url "https://<your-ngrok-domain>"
```

For PubMed fetch check include email (recommended) and keep `max_results` small:

```bash
python scripts/colab_smoke_check.py \
  --base-url "https://<your-ngrok-domain>" \
  --email your_email@example.edu
```

If your Colab runtime blocks external NCBI requests, skip this step:

```bash
python scripts/colab_smoke_check.py \
  --base-url "https://<your-ngrok-domain>" \
  --skip-pubmed
```

### Colab notebook smoke snippet

You can paste this directly in Colab as a cell (after `BASE_URL` is set):

```python
import os
import requests

base = BASE_URL.rstrip("/")
probe_email = "you@example.edu"  # optional when skipping pubmed

print("health", requests.get(f"{base}/health", timeout=20).status_code)
pr = requests.post(f"{base}/projects", json={"name":"Colab smoke","disease_key":"colab","description":"smoke"}, timeout=20)
print("create project", pr.status_code, pr.text[:200])
pid = pr.json()["id"]

cmp = requests.post(
    f"{base}/projects/{pid}/data/compare/litsuggest",
    json={
        "primary":[{"pmid":"10","label":1},{"pmid":"11","label":0}],
        "litsuggest":[{"pmid":"10","score":0.87},{"pmid":"11","score":0.15}],
        "score_threshold":0.5,
    },
    timeout=20,
)
print("compare", cmp.status_code, cmp.json())

fetch = requests.post(
    f"{base}/projects/{pid}/data/pubmed/fetch",
    json={"email":probe_email, "query": '\"diabetic kidney disease\"[Title/Abstract]', "max_results":5},
    timeout=60,
)
print("pubmed fetch", fetch.status_code)
print(fetch.json() if fetch.headers.get("content-type","").startswith("application/json") else fetch.text[:200])
```

### Docker deployment (recommended)

Set token before compose (optional, but recommended):

```bash
export HF_TOKEN=your_hf_token_here
```

For HTTPS on `l2g.aiteb.app`, create `.env` and set these values:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
SSL_DOMAIN=l2g.aiteb.app
SSL_DOMAIN_ALIASES=www.l2g.aiteb.app
SSL_EMAIL=ops@l2g.aiteb.app
```

Run one of:

```bash
# First issue/renew certs
CERTBOT_MODE=webroot ./scripts/ssl_manage.sh ssl-manual

# If you are not yet serving HTTP yet, use standalone mode
# (the script will temporarily stop/start the gateway when needed):
CERTBOT_MODE=standalone ./scripts/ssl_manage.sh ssl-manual
# or
./scripts/ssl_manage.sh ssl-manual

# Optional: enable cron-based auto renewal
./scripts/ssl_manage.sh ssl-setup-cron
```

If this is the first deployment and `/etc/letsencrypt/live/<domain>/` is not present yet, do this order:

```bash
# 1) (Re)build gateway image once after code updates
docker compose build gateway

# 2) Start gateway in HTTP bootstrap mode (it will switch to HTTPS after cert exists)
docker compose up -d gateway

# 3) Run SSL issuance, then verify cert files exist
CERTBOT_MODE=webroot ./scripts/ssl_manage.sh ssl-manual
# Check:
ls -l /etc/letsencrypt/live/l2g.aiteb.app/fullchain.pem
```

Then launch the full stack:

```bash
docker compose up --build -d
```

If certbot still returns `404` for ACME paths, validate gateway challenge routing:

```bash
# Create a test file inside the gateway webroot and verify it is reachable
docker exec -it l2g-gateway sh -c "mkdir -p /var/www/certbot/.well-known/acme-challenge && echo ok > /var/www/certbot/.well-known/acme-challenge/test-gateway"
curl -i http://l2g.aiteb.app/.well-known/acme-challenge/test-gateway
```

Expected output: `HTTP/1.1 200 OK` with body `ok`.

Open:

- App UI: [https://l2g.aiteb.app](https://l2g.aiteb.app) (gateway proxy on 443)
- API docs: [https://l2g.aiteb.app/api/docs](https://l2g.aiteb.app/api/docs)
- Health check: [http://localhost:8000/health](http://localhost:8000/health)
- Gateway health check: [https://l2g.aiteb.app/health](https://l2g.aiteb.app/health)

This compose stack now uses:

- Starts `backend` (FastAPI) on port `8000`.
- Builds and serves `frontend` with Nginx (internal service on port `80` only).
- Runs a dedicated `gateway` container on host ports `80/443` for HTTPS and public routing.
- Proxies API calls from the UI at `/api/*` to `backend:8000`.

You can find deployment files here:

- `backend/Dockerfile`
- `frontend/Dockerfile`
- `frontend/nginx.conf`
- `gateway/Dockerfile`
- `gateway/nginx.conf`
- `gateway/default.conf.template`
- `scripts/ssl_manage.sh`
- `docker-compose.yml`
- `.dockerignore`

If you want a custom API base at build time:

```bash
docker compose build --build-arg VITE_API_BASE=https://api.example.com frontend
docker compose up frontend
```

Project artifacts are persisted in `${HOME}/l2g-data` (mounted to `/app/data`) and `${HOME}/.cache/huggingface` in `docker-compose.yml`.

Before first run, ensure host paths exist and are writable:

```bash
mkdir -p "${HOME}/l2g-data" "${HOME}/.cache/huggingface"
```

You can run this helper before starting services:

```bash
./scripts/ensure_storage_paths.sh
```

Optional overrides:

```bash
L2G_DATA_DIR=/path/to/storage ./scripts/ensure_storage_paths.sh
HF_CACHE_DIR=/path/to/hf-cache ./scripts/ensure_storage_paths.sh
```

## Core library

Importable Python package `l2g_core` (installed with `pip install -e .`):

- `l2g_core.devices`: `resolve_torch_device("cuda"|"mps"|"cpu"|"auto")`
- `l2g_core.relevance`: train + predict BioBERT classifier
- `l2g_core.ner`: Hugging Face NER pipeline (default BENT-PubMedBERT)
- `l2g_core.normalization`: character rules + MyGene + optional Wikipedia
- `l2g_core.pipeline`: orchestration helpers

## API overview

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/devices` | CUDA/MPS availability |
| GET/POST | `/projects` | List/create projects |
| GET | `/projects/{id}/models` | List trained model folders |
| POST | `/train/{id}/relevance` | Start training job (background) |
| POST | `/train/{id}/relevance/kfold` | Start k-fold job |
| GET | `/train/jobs` | List recent jobs (optional `project_id` filter) |
| GET | `/train/jobs` | List recent jobs (optional `project_id`, `limit`) |
| GET | `/train/jobs/{job_id}` | Poll job status + metrics |
| POST | `/pipeline/run` | Run classify / ner / normalize / full |
| GET | `/projects/{id}/data/last-run` | List files in `outputs/last_run` |
| POST | `/projects/{id}/data/import/articles` | `multipart/form-data` file → JSON articles |
| POST | `/projects/{id}/data/import/mentions` | File → JSON mention rows (normalize step) |
| GET | `/projects/{id}/data/export/{artifact}?format=` | `artifact` = `classification`, `mentions`, `normalized` (`csv` / `xlsx` / `pkl`) |
| GET | `/projects/{id}/data/export/bundle?format=` | `pkl` or `xlsx` (all tables present on disk) |
| GET | `/projects/{id}/data/templates/articles` | Download CSV column template |
| GET | `/projects/{id}/data/templates/mentions` | Mentions CSV template |
| POST | `/projects/{id}/data/pubmed/fetch` | Fetch abstracts from PubMed via Entrez |
| POST | `/projects/{id}/data/import/litsuggest-scores` | Upload LitSuggest score rows (`pmid`,`score`) |
| POST | `/projects/{id}/data/compare/litsuggest` | Compare model labels against LitSuggest scores |

## Notes

- **NER training** is not fine-tuned in this stack; the default is a pretrained `pruas/BENT-PubMedBERT-NER-Gene` model. You can swap `ner_model` in API/UI.
- **NER methods** now support both `transformers` (default, default HF pipeline) and optional legacy `bent` extraction. In the UI, enable **Compare methods** in Pipeline to run both and inspect mention overlap.
- **Bent method** is optional and uses the legacy release `bent==0.0.80` (latest release compatible with Python 3.10.13).
  - use `./scripts/setup_bent_runtime.sh` for local Python install (separate `.venv-bent` runtime)
  - for runtime integration set `BENT_SERVICE_URL` and point it to your bent service:
    - local service: `http://127.0.0.1:8010`
    - compose service: `http://bent-runtime:8010`
    - helper: `./scripts/run_bent_service.sh` or `./scripts/setup_docker_bent.sh`
- **Normalization** uses live **MyGene** queries; rate-limiting sleep is applied. For high volume, add local caching (future improvement).
- **Jobs** are stored in memory; use Redis/Queue for multi-worker production.

## License

Use under your team’s policy; add a `LICENSE` file if you distribute publicly.
