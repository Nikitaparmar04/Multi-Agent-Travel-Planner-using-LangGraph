<div align="center">

<h1>✈️ TripMind — Multi-Agent AI Travel Planner</h1>

<p>
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-ReAct_Agent-FF6B6B?style=for-the-badge&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-LLaMA_3-F55036?style=for-the-badge&logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-o4--mini-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/LLMOps-Production_Ready-00C853?style=for-the-badge&logo=kubernetes&logoColor=white" />
</p>

<p><strong>A production-grade, multi-agent AI travel planning system powered by LangGraph ReAct agents, FastAPI backend, and a stunning glassmorphism web UI. Plan any trip worldwide with real-time itineraries, weather, currency conversion, and cost breakdowns — all in one shot.</strong></p>

<br/>

![TripMind Demo](https://img.shields.io/badge/Live_Demo-Try_Now-blueviolet?style=for-the-badge)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-reference)
- [LLMOps & Deployment](#-llmops--deployment)
  - [Local Development](#1-local-development)
  - [Docker Deployment](#2-docker-deployment)
  - [Cloud Deployment (AWS / GCP / Azure)](#3-cloud-deployment)
  - [CI/CD Pipeline](#4-cicd-pipeline)
  - [Monitoring & Observability](#5-monitoring--observability)
  - [LLM Cost Management](#6-llm-cost-management)
  - [Prompt Versioning](#7-prompt-versioning)
- [Configuration](#-configuration)
- [Extending the Agent](#-extending-the-agent)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌍 Overview

**TripMind** is an intelligent, multi-agent travel planning assistant built with the **LangGraph ReAct (Reasoning + Acting)** framework. Unlike simple chatbots, TripMind orchestrates a graph of specialized agents that can autonomously reason, call tools, and synthesize real-world data into a complete, personalised travel plan.

### What TripMind Can Do

| Capability | Description |
|---|---|
| 🗺️ **Itinerary Planning** | Day-by-day trip plans for any destination worldwide |
| 🏨 **Hotel Recommendations** | Curated accommodation options with per-night pricing |
| 🌤️ **Real-time Weather** | Live weather data and seasonal advice |
| 💱 **Currency Conversion** | Live exchange rates for budget planning |
| 🍽️ **Restaurant Discovery** | Local dining recommendations with price ranges |
| 🎯 **Dual Itineraries** | Mainstream tourist + hidden off-beat routes |
| 💰 **Cost Breakdown** | Detailed expense budgets per day and total trip |
| 🚌 **Transport Info** | Local transport modes, costs, and logistics |

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TripMind System                             │
│                                                                     │
│  ┌──────────────┐   HTTP/REST    ┌──────────────────────────────┐  │
│  │              │ ─────────────► │        FastAPI Backend        │  │
│  │  Web Frontend│                │  (app.py · /query endpoint)  │  │
│  │  (HTML/CSS/  │ ◄───────────── │                              │  │
│  │   Vanilla JS)│   JSON         └──────────────┬───────────────┘  │
│  └──────────────┘                               │                  │
│                                                 ▼                  │
│                                  ┌──────────────────────────────┐  │
│                                  │     LangGraph State Machine   │  │
│                                  │                              │  │
│                                  │  START ──► [Agent Node]      │  │
│                                  │              │               │  │
│                                  │        tools_condition       │  │
│                                  │           ┌──┴──┐            │  │
│                                  │          ▼      ▼           │  │
│                                  │     [Tools]    END           │  │
│                                  │      Node                   │  │
│                                  │        │                    │  │
│                                  │        └────► [Agent Node]  │  │
│                                  └──────────────────────────────┘  │
│                                                 │                  │
│                              ┌──────────────────┼──────────────┐  │
│                              │    Tool Layer     │              │  │
│                              │                  ▼              │  │
│                              │  ┌─────────┐ ┌────────┐        │  │
│                              │  │ Weather │ │Currency│ ....   │  │
│                              │  │  Tool   │ │  Tool  │        │  │
│                              │  └─────────┘ └────────┘        │  │
│                              └──────────────────────────────────┘  │
│                                                 │                  │
│                              ┌──────────────────┼──────────────┐  │
│                              │      LLM Layer                   │  │
│                              │                                  │  │
│                              │  ┌───────────┐ ┌──────────────┐ │  │
│                              │  │   Groq    │ │   OpenAI     │ │  │
│                              │  │ LLaMA 3 / │ │   o4-mini    │ │  │
│                              │  │DeepSeek R1│ │              │ │  │
│                              │  └───────────┘ └──────────────┘ │  │
│                              └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Agent Flow (LangGraph ReAct)

```
User Query
    │
    ▼
[Agent Node] ── Reasons with LLM + System Prompt
    │
    ├── Needs tool? ─► [Tool Node] ── Executes tool(s)
    │                       │
    │                       └────────────► [Agent Node] (loop)
    │
    └── Done? ──────────────────────────► [END] ── Return Response
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM Orchestration** | LangGraph | ReAct agent graph & state management |
| **LLM Providers** | Groq (DeepSeek R1 / LLaMA 3), OpenAI (o4-mini) | Language model inference |
| **LLM Framework** | LangChain | Prompt templating, tool binding, message types |
| **Backend API** | FastAPI + Uvicorn | High-performance async REST API |
| **Frontend** | HTML5 + Vanilla CSS + JavaScript | Glassmorphism chat UI |
| **Config** | YAML + python-dotenv | Environment & model configuration |
| **Packaging** | pyproject.toml + uv | Python dependency management |
| **Tools** | Custom LangChain tools | Weather, currency, place search, calculator |

---

## 📁 Project Structure

```
Multi-Agent_Trip_Planeer/
│
├── 📄 app.py                    # FastAPI application & /query endpoint
├── 📄 main.py                   # Package entry point
├── 📄 pyproject.toml            # Project metadata & dependencies
├── 📄 requirements.txt          # pip-compatible dependency list
├── 📄 setup.py                  # Editable install config
├── 📄 .env                      # API keys (git-ignored)
├── 📄 .gitignore
│
├── 🤖 agent/
│   └── agentic_workflow.py      # LangGraph GraphBuilder (ReAct loop)
│
├── 🔧 tools/
│   ├── __init__.py
│   ├── place_search_tool.py     # Google Places / search tool
│   ├── whether_info_tool.py     # Real-time weather API tool
│   ├── currency_converter_tool.py  # Live FX rates tool
│   └── calculator_tool.py       # Math/budget calculator tool
│
├── 💬 prompt_library/
│   └── prompt.py                # System prompt (versioned)
│
├── ⚙️ config/
│   └── config.yaml              # LLM provider & model config
│
├── 🔨 utils/
│   ├── __init__.py
│   ├── model_loader.py          # LLM factory (Groq / OpenAI)
│   └── config_loader.py         # YAML config parser
│
├── 🌐 frontend/
│   ├── index.html               # App shell & chat UI
│   ├── style.css                # Glassmorphism design system
│   └── app.js                   # Chat logic, markdown rendering
│
└── 📓 notebook/                 # Jupyter exploration notebooks
```

---

## ✨ Features

### 🤖 Multi-Agent Intelligence
- **LangGraph ReAct loop** — agent autonomously decides when to call tools and when to finalize
- **Tool chaining** — multiple tools can be called in sequence within a single user query
- **Stateful messages** — full conversation context maintained via `MessagesState`

### 🎨 Premium Chat UI
- Glassmorphism dark-mode design with animated background orbs
- Sidebar with trip history and model status badge
- Suggestion cards for instant query inspiration
- Markdown rendering for structured travel plans
- Mobile-responsive with collapsible sidebar

### 🔀 Multi-Provider LLM Support
- Switch between **Groq** (ultra-fast, free tier) and **OpenAI** (powerful reasoning)
- Configured via `config/config.yaml` — no code changes needed
- Currently supports: `deepseek-r1-distill-llama-70b` (Groq) and `o4-mini` (OpenAI)

### 🛡️ Production-Ready Backend
- FastAPI with CORS middleware
- Pydantic request validation
- Graceful error handling with proper HTTP status codes
- Health check endpoint at `GET /`

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.12 |
| pip / uv | Latest |
| Groq API Key | [Get free key](https://console.groq.com) |
| OpenAI API Key | [Optional](https://platform.openai.com) |

### 1. Clone the Repository

```bash
git clone https://github.com/Nikitaparmar04/Multi-Agent-Travel-Planner-using-LangGraph.git
cd Multi-Agent-Travel-Planner-using-LangGraph
```

### 2. Create Virtual Environment

```bash
# Using uv (recommended — much faster)
pip install uv
uv venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
# Using uv
uv pip install -r requirements.txt

# OR using pip
pip install -r requirements.txt
pip install -e .                 # Editable install (for local package)
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
# LLM Provider API Keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here        # optional

# Tool API Keys
TAVILY_API_KEY=your_tavily_api_key_here        # for web search
OPENWEATHER_API_KEY=your_openweather_key_here  # for weather tool
```

### 5. Choose Your LLM Provider

Edit `config/config.yaml`:

```yaml
llm:
  openai:
    provider: "openai"
    model_name: "o4-mini"
  groq:
    provider: "groq"
    model_name: "deepseek-r1-distill-llama-70b"
```

### 6. Start the Backend

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 7. Open the Frontend

Open `frontend/index.html` directly in your browser, **or** serve it:

```bash
# Python simple server
cd frontend
python -m http.server 3000
# Then visit: http://localhost:3000
```

> ✅ The frontend talks to the backend at `http://localhost:8000` by default.

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes (if using Groq) | Groq Cloud API key |
| `OPENAI_API_KEY` | ✅ Yes (if using OpenAI) | OpenAI platform API key |
| `TAVILY_API_KEY` | 🔶 Recommended | Web search tool (free tier available) |
| `OPENWEATHER_API_KEY` | 🔶 Recommended | Weather data API |

---

## 📡 API Reference

### `GET /`
Health check.

**Response:**
```json
{
  "status": "ok",
  "service": "TripMind API"
}
```

---

### `POST /query`
Submit a travel planning query to the multi-agent system.

**Request Body:**
```json
{
  "query": "Plan a 7-day trip to Japan with budget breakdown"
}
```

**Success Response (`200 OK`):**
```json
{
  "answer": "# 🗾 7-Day Japan Itinerary\n\n## Day 1 — Tokyo Arrival\n..."
}
```

**Error Response (`500 Internal Server Error`):**
```json
{
  "error": "Description of the error"
}
```

---

## 🏭 LLMOps & Deployment

This section covers everything needed to take TripMind from local development to a production-grade, observable, cost-efficient deployment.

---

### 1. Local Development

```bash
# Hot-reload mode — backend auto-restarts on file changes
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run with verbose LangChain tracing
LANGCHAIN_TRACING_V2=true uvicorn app:app --reload
```

#### Enable LangSmith Tracing (Highly Recommended)

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=tripmind-dev
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

All LangGraph traces will appear in [LangSmith](https://smith.langchain.com) — inspect every agent step, tool call, token usage, and latency.

---

### 2. Docker Deployment

#### Create `Dockerfile`

```dockerfile
# ─── Stage 1: Builder ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app
COPY requirements.txt pyproject.toml setup.py ./
RUN pip install --no-cache-dir uv && \
    uv pip install --system -r requirements.txt

# ─── Stage 2: Runtime ───────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

#### Create `docker-compose.yml`

```yaml
version: "3.9"

services:
  tripmind-api:
    build: .
    container_name: tripmind_backend
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY}
      - LANGCHAIN_TRACING_V2=${LANGCHAIN_TRACING_V2}
      - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
      - LANGCHAIN_PROJECT=tripmind-prod
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  tripmind-frontend:
    image: nginx:alpine
    container_name: tripmind_frontend
    ports:
      - "3000:80"
    volumes:
      - ./frontend:/usr/share/nginx/html:ro
    depends_on:
      - tripmind-api
    restart: unless-stopped
```

#### Build & Run

```bash
# Build image
docker build -t tripmind:latest .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f tripmind-api

# Stop
docker-compose down
```

---

### 3. Cloud Deployment

#### Option A — Deploy to Railway (Easiest)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login & deploy
railway login
railway init
railway up

# Set environment variables
railway variables set GROQ_API_KEY=your_key
railway variables set OPENAI_API_KEY=your_key
```

#### Option B — Deploy to AWS EC2

```bash
# 1. SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 2. Install dependencies
sudo apt update && sudo apt install -y python3.12 python3-pip git docker.io
sudo systemctl start docker

# 3. Clone & deploy
git clone https://github.com/Nikitaparmar04/Multi-Agent-Travel-Planner-using-LangGraph.git
cd Multi-Agent-Travel-Planner-using-LangGraph

# 4. Set environment variables
cp .env.example .env
nano .env  # Add your API keys

# 5. Build & run with Docker
docker-compose up -d --build
```

#### Option C — Deploy to Google Cloud Run

```bash
# 1. Authenticate
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Build & push Docker image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/tripmind

# 3. Deploy to Cloud Run
gcloud run deploy tripmind \
  --image gcr.io/YOUR_PROJECT_ID/tripmind \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GROQ_API_KEY=your_key,OPENAI_API_KEY=your_key \
  --memory 512Mi \
  --concurrency 10
```

#### Option D — Deploy to Azure Container Apps

```bash
# 1. Login to Azure
az login
az group create --name tripmind-rg --location eastus

# 2. Create Container Registry
az acr create --name tripmindacr --resource-group tripmind-rg --sku Basic
az acr build --registry tripmindacr --image tripmind:latest .

# 3. Deploy Container App
az containerapp create \
  --name tripmind \
  --resource-group tripmind-rg \
  --image tripmindacr.azurecr.io/tripmind:latest \
  --target-port 8000 \
  --ingress external \
  --env-vars GROQ_API_KEY=secretref:groq-key
```

---

### 4. CI/CD Pipeline

#### GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: TripMind CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  # ── Lint & Test ─────────────────────────────────────────────────────
  test:
    name: Lint & Test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.12
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: |
          uv venv
          source .venv/bin/activate
          uv pip install -r requirements.txt
          uv pip install pytest ruff

      - name: Lint with ruff
        run: ruff check .

      - name: Run tests
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          source .venv/bin/activate
          pytest tests/ -v

  # ── Build & Push Docker Image ────────────────────────────────────────
  build:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t tripmind:${{ github.sha }} .

      - name: Push to Docker Hub
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker tag tripmind:${{ github.sha }} ${{ secrets.DOCKER_USERNAME }}/tripmind:latest
          docker push ${{ secrets.DOCKER_USERNAME }}/tripmind:latest

  # ── Deploy ───────────────────────────────────────────────────────────
  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy via SSH
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.SERVER_HOST }}
          username: ${{ secrets.SERVER_USER }}
          key: ${{ secrets.SSH_PRIVATE_KEY }}
          script: |
            cd /opt/tripmind
            docker-compose pull
            docker-compose up -d --build
            docker system prune -f
```

---

### 5. Monitoring & Observability

#### LangSmith (LLM Tracing)

Add to `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_langsmith_key
LANGCHAIN_PROJECT=tripmind-production
```

This gives you:
- ✅ Full agent trace visualization
- ✅ Per-run token usage & cost
- ✅ Tool call success/failure rates
- ✅ Latency histograms per node
- ✅ Prompt version comparison

#### FastAPI Metrics (Prometheus + Grafana)

Install:

```bash
pip install prometheus-fastapi-instrumentator
```

Add to `app.py`:

```python
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

Then scrape `/metrics` with Prometheus and visualize in Grafana.

#### Logging

TripMind uses Python's standard logging. In production, pipe logs to:

- **AWS CloudWatch** — `pip install watchtower`
- **Datadog** — `pip install ddtrace`
- **Structured JSON** — `pip install python-json-logger`

Example structured logging setup:

```python
import logging
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

---

### 6. LLM Cost Management

| Strategy | Implementation |
|---|---|
| **Use Groq for dev/test** | Free tier, ultra-fast inference |
| **Use OpenAI o4-mini in prod** | Best quality/cost ratio |
| **Cache frequent responses** | Add Redis caching on `/query` |
| **Token limits** | Set `max_tokens` in model config |
| **LangSmith budget alerts** | Set monthly spend alerts |

#### Add Response Caching (Redis)

```bash
pip install redis fastapi-cache2
```

```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

@app.on_event("startup")
async def startup():
    FastAPICache.init(RedisBackend("redis://localhost:6379"), prefix="tripmind")

@app.post("/query")
@cache(expire=3600)  # cache for 1 hour
async def query_travel_agent(request: QueryRequest):
    ...
```

---

### 7. Prompt Versioning

Prompts are stored in `prompt_library/prompt.py`. Follow these practices:

```
prompt_library/
├── prompt.py            ← current production prompt (imported by agent)
├── v1_prompt.py         ← baseline prompt
├── v2_prompt.py         ← itinerary improvements
└── experiments/
    └── cot_prompt.py    ← chain-of-thought variant
```

**Best practices:**
- Commit every prompt change with a descriptive message
- A/B test prompts using LangSmith's [Playground](https://smith.langchain.com)
- Track prompt performance (quality, token count, user satisfaction)
- Never edit the production prompt without a comparison run

---

## ⚙️ Configuration

### `config/config.yaml`

```yaml
llm:
  openai:
    provider: "openai"
    model_name: "o4-mini"       # Change to gpt-4o, gpt-4-turbo etc.
  groq:
    provider: "groq"
    model_name: "deepseek-r1-distill-llama-70b"   # Or llama-3.1-70b-versatile
```

### Switching LLM Provider

In `utils/model_loader.py`, the `ModelLoader` class accepts a `model_provider` parameter:

```python
# Use Groq (default, fast & free)
loader = ModelLoader(model_provider="groq")
llm = loader.load_llm()

# Use OpenAI
loader = ModelLoader(model_provider="openai")
llm = loader.load_llm()
```

---

## 🔌 Extending the Agent

### Adding a New Tool

1. Create a new file in `tools/`:

```python
# tools/hotel_search_tool.py
from langchain.tools import tool

@tool
def hotel_search(destination: str, check_in: str, check_out: str) -> str:
    """Search for available hotels at a destination for given dates."""
    # Your implementation here (e.g., call Hotels.com API)
    return f"Found 10 hotels in {destination} for {check_in} to {check_out}..."
```

2. Register the tool in `agent/agentic_workflow.py`:

```python
from tools.hotel_search_tool import hotel_search
from tools.whether_info_tool import get_weather

class GraphBuilder():
    def __init__(self):
        self.tools = [
            hotel_search,
            get_weather,
            # add more tools here
        ]
```

### Adding a New LLM Provider

Extend `utils/model_loader.py`:

```python
elif self.model_provider == "anthropic":
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** your changes: `git commit -m "feat: add hotel search tool"`
4. **Push** to your branch: `git push origin feature/my-feature`
5. **Open** a Pull Request

### Commit Convention

```
feat:     New feature
fix:      Bug fix
docs:     Documentation update
refactor: Code refactor
perf:     Performance improvement
test:     Tests added/updated
chore:    Tooling, config, dependencies
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with ❤️ using LangGraph · FastAPI · Groq · OpenAI**

⭐ **Star this repo if TripMind helped you plan your next adventure!** ⭐

</div>
