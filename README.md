# Research Assistant

Plan -> search -> read -> extract facts -> gap check -> write -> fact-check, streamed live to the browser.

## Run
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env     # add ANTHROPIC_API_KEY and TAVILY_API_KEY (free tier at tavily.com)
    cd backend && uvicorn main:app --reload
Open http://localhost:8000

## Docker
    docker build -t research-assistant .
    docker run -p 8000:8000 --env-file .env research-assistant

## Why the output is accurate
- Structured outputs via forced tool calls (no JSON parsing errors)
- Per-source fact extraction; the writer sees only extracted facts, never raw pages
- Sources rated by quality; the writer prefers high-quality ones and flags conflicts
- Gap-check loop runs follow-up searches when evidence is thin
- A separate fact-check pass finds unsupported claims and triggers a rewrite
- Citations pointing at non-existent sources are stripped in code
- Prompt-injection guard on all web content, SSRF protection on page fetches
- Retries, timeouts, iteration caps, and a concurrency limit keep cost and failures bounded

## Ideas to extend
Add auth, pgvector search over past reports, an eval set of 20-30 questions, and per-report cost tracking.
