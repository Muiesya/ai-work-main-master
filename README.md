# AI Drug Q&A (Python)

A minimal retrieval-augmented medication information Q&A service built with Python and FastAPI. Uses a small sample knowledge base and the DeepSeek API (OpenAI-compatible) for grounded answers with safety messaging.

## Features
- Simple knowledge base (`data/drug_facts.json`) with sample drug entries.
- TF-IDF retriever to surface relevant drug records for grounding.
- DeepSeek-backed generation with safety-focused system prompt and disclaimer.
- FastAPI service exposing `/ask` and `/health` endpoints.

## Requirements
- Python 3.11+
- DeepSeek API key exported as `DEEPSEEK_API_KEY`.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DEEPSEEK_API_KEY="your-key"
uvicorn src.main:app --reload
```

If you are using a custom DeepSeek-compatible endpoint, set `DEEPSEEK_API_KEY` and optionally override the `api_base_url` default (`https://api.deepseek.com`) in `src/app/config.py`.

## Usage
Send a POST request to `/ask` with a question:
```bash
curl -X POST "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "对乙酰氨基酚的常见副作用是什么？"}'
```

Example response:
```json
{
  "question": "对乙酰氨基酚的常见副作用是什么？",
  "answer": "...模型生成的简短说明，并包含免责声明...",
  "sources": ["acetaminophen (brands: Tylenol)"],
  "last_updated": ["2024-06-01"]
}
```

### Simple Web UI
- After starting the server, open http://127.0.0.1:8000/ to use the minimal frontend.
- Enter a question, click “询问” to send it to the API, and view the grounded answer plus sources and update dates.

## Safety Notes
- The service is for general medication information only and does **not** provide personalized medical advice or dosing instructions.
- Always consult a licensed clinician or pharmacist before making medication decisions.
- Data should be refreshed from authoritative sources (e.g., FDA labels, clinical guidelines) and audited regularly.
