# 🤝 Personalized Networking Assistant

> An AI-powered web application that helps users prepare for networking events by analyzing event descriptions, extracting themes, generating personalized conversation starters, and verifying facts.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Example Requests](#example-requests)
- [Future Improvements](#future-improvements)

---

## Overview

The **Personalized Networking Assistant** uses a combination of **DistilBERT zero-shot classification** and **GPT-2 text generation** to:

1. **Analyze** event descriptions and extract key themes (AI, Climate, Healthcare, etc.)
2. **Generate** 3 personalized conversation starters tailored to your interests
3. **Fact-check** topics against Wikipedia
4. **Store** all conversations in `data/history.json`
5. **Accept and persist** user feedback in `data/feedback.json`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                           │
│  (streamlit_ui.py) – UI only, communicates via HTTPX            │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP (port 8000)
┌───────────────────────────▼─────────────────────────────────────┐
│                     FastAPI Backend                              │
│  backend/main.py → routes.py → schemas.py                       │
└──┬──────────────────┬───────────────────────┬───────────────────┘
   │                  │                       │
   ▼                  ▼                       ▼
services/         services/             services/
event_analyzer   topic_generator        fact_checker
(DistilBERT       (GPT-2 Small)         (Wikipedia API)
 Zero-Shot)
   │                                        │
   └──────────────────────────────────────┐ │
                                          ▼ ▼
                                   data/history.json
                                   data/feedback.json
```

---

## Features

| Feature | Description |
|---|---|
| 🧠 **Event Analysis** | Zero-shot classification extracts up to 5 themes from any event description |
| 💬 **Conversation Starters** | GPT-2 generates 3 personalized, unique networking openers |
| 📚 **Fact Check** | Wikipedia API summarizes any topic in 3 sentences |
| 📜 **History** | Every conversation is persisted with full metadata |
| ⭐ **Favorites** | Mark and filter favorite conversations |
| 💬 **Feedback** | Thumbs-up/down + comments stored per conversation |
| 🔒 **Validation** | Full Pydantic v2 request/response validation |
| 🛡️ **Error Handling** | Graceful handling of missing pages, GPU errors, timeouts |

---

## Project Structure

```
personalized-networking-assistant/
│
├── app.py                    # Unified launcher
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Pytest configuration
├── README.md
├── .gitignore
│
├── backend/
│   ├── __init__.py
│   ├── main.py               # FastAPI app factory
│   ├── routes.py             # All route handlers
│   ├── schemas.py            # Pydantic request/response models
│   └── config.py             # Centralized configuration
│
├── services/
│   ├── __init__.py
│   ├── event_analyzer.py     # DistilBERT zero-shot classification
│   ├── topic_generator.py    # GPT-2 conversation starter generation
│   ├── fact_checker.py       # Wikipedia fact verification
│   ├── history_logger.py     # JSON history persistence
│   └── feedback_logger.py    # JSON feedback persistence
│
├── frontend/
│   ├── __init__.py
│   └── streamlit_ui.py       # Full Streamlit UI
│
├── data/
│   ├── history.json          # Conversation history store
│   └── feedback.json         # User feedback store
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures
│   ├── test_event_analyzer.py
│   ├── test_topic_generator.py
│   ├── test_fact_checker.py
│   └── test_routes.py
│
└── models/                   # (auto-populated by HuggingFace)
    └── README.md
```

---

## Installation

### Prerequisites

- Python 3.11+
- pip
- (Optional) NVIDIA GPU with CUDA for faster inference

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/personalized-networking-assistant.git
cd personalized-networking-assistant
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Pre-download Models

First-time model downloads can take a few minutes. Pre-download them:

```bash
python -c "from transformers import pipeline; pipeline('zero-shot-classification', model='cross-encoder/nli-distilroberta-base')"
python -c "from transformers import pipeline; pipeline('text-generation', model='gpt2')"
```

### 5. Environment Variables (Optional)

Create a `.env` file to customize settings:

```env
API_HOST=127.0.0.1
API_PORT=8000
ZERO_SHOT_MODEL=cross-encoder/nli-distilroberta-base
TEXT_GEN_MODEL=gpt2
MAX_NEW_TOKENS=80
TEMPERATURE=0.8
TOP_P=0.95
NUM_STARTERS=3
LOG_LEVEL=INFO
```

---

## Running the Application

Open **two separate terminals**.

### Terminal 1 – Start FastAPI Backend

```bash
uvicorn backend.main:app --reload
```

Backend runs at: **http://127.0.0.1:8000**  
API Docs: **http://127.0.0.1:8000/docs**

### Terminal 2 – Start Streamlit Frontend

```bash
streamlit run frontend/streamlit_ui.py
```

Frontend runs at: **http://localhost:8501**

### Alternative: Use the Launcher

```bash
# Start backend
python app.py --mode backend

# Start frontend (in a second terminal)
python app.py --mode frontend
```

---

## API Documentation

Interactive Swagger UI: **http://127.0.0.1:8000/docs**  
ReDoc: **http://127.0.0.1:8000/redoc**

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/v1/health` | Liveness probe |
| POST | `/api/v1/analyze-event` | Extract themes from event description |
| POST | `/api/v1/generate-conversation` | Generate personalized conversation starters |
| GET | `/api/v1/fact-check?topic=...` | Verify topic via Wikipedia |
| GET | `/api/v1/history` | Retrieve conversation history |
| POST | `/api/v1/feedback` | Submit feedback for a conversation |
| GET | `/api/v1/feedback` | Retrieve all feedback |

---

## Example Requests

### Generate Conversation Starters

```bash
curl -X POST http://127.0.0.1:8000/api/v1/generate-conversation \
  -H "Content-Type: application/json" \
  -d '{
    "event_description": "AI for Sustainable Cities summit bringing together tech leaders and urban planners",
    "user_interests": ["climate change", "urban planning"]
  }'
```

**Response:**

```json
{
  "conversation_id": "a3f4c1b2-...",
  "themes": ["AI", "Urban Planning", "Sustainability"],
  "scores": [0.9234, 0.8876, 0.7654],
  "starters": [
    "How do you see AI helping cities reduce emissions?",
    "I'm interested in sustainable infrastructure. Have you worked on similar projects?",
    "What emerging AI technologies do you think will transform urban planning?"
  ],
  "timestamp": "2026-07-11T15:00:00Z"
}
```

### Fact Check

```bash
curl "http://127.0.0.1:8000/api/v1/fact-check?topic=Blockchain+in+Healthcare"
```

**Response:**

```json
{
  "topic": "Blockchain in Healthcare",
  "title": "Blockchain",
  "summary": "A blockchain is a distributed ledger with growing lists of records. It was invented by Satoshi Nakamoto. It is used across industries including healthcare.",
  "verified": true,
  "url": "https://en.wikipedia.org/wiki/Blockchain"
}
```

### Submit Feedback

```bash
curl -X POST http://127.0.0.1:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "a3f4c1b2-...",
    "thumbs_up": true,
    "comment": "Really helpful starters!"
  }'
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_routes.py -v

# Run tests matching a pattern
pytest -k "test_happy_path" -v
```

### Test Coverage Targets

| Module | Coverage Target |
|---|---|
| `services/event_analyzer.py` | ≥ 90% |
| `services/topic_generator.py` | ≥ 90% |
| `services/fact_checker.py` | ≥ 90% |
| `backend/routes.py` | ≥ 90% |

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `ZERO_SHOT_MODEL` | `cross-encoder/nli-distilroberta-base` | HuggingFace model for event analysis |
| `TEXT_GEN_MODEL` | `gpt2` | HuggingFace model for text generation |
| `MAX_NEW_TOKENS` | `80` | Max tokens per generated starter |
| `TEMPERATURE` | `0.8` | Sampling temperature (creativity) |
| `TOP_P` | `0.95` | Nucleus sampling threshold |
| `NUM_STARTERS` | `3` | Number of starters to generate |
| `WIKIPEDIA_LANGUAGE` | `en` | Wikipedia language |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Future Improvements

- [ ] **User Authentication** – JWT-based login and user profiles
- [ ] **Vector Search** – Semantic history search using FAISS or ChromaDB
- [ ] **LLM Upgrade** – Replace GPT-2 with a more capable model (e.g., Mistral 7B)
- [ ] **Database Backend** – Replace JSON files with PostgreSQL via SQLAlchemy
- [ ] **Docker Compose** – Containerized multi-service deployment
- [ ] **CI/CD Pipeline** – GitHub Actions with automated testing and coverage
- [ ] **Export Feature** – Download conversation starters as PDF or Markdown
- [ ] **Multi-language Support** – Wikipedia lookups in multiple languages
- [ ] **Real-time Suggestions** – WebSocket-based streaming of starter tokens
- [ ] **Analytics Dashboard** – Visualize theme trends and feedback stats

---

## License

MIT License – see [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request
