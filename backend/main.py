import asyncio, json, os, sqlite3, time, traceback, uuid
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # must run before importing agent so env vars are set

import openai
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import agent
from agent import research

DB = os.getenv("DB_PATH", "reports.db")
db = sqlite3.connect(DB, check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS reports(id TEXT PRIMARY KEY, question TEXT, report TEXT, "
           "sources TEXT, meta TEXT, created REAL)")

app = FastAPI(title="Research Assistant")
slots = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT", "2")))  # cap parallel runs


class Ask(BaseModel):
    question: str = Field(min_length=5, max_length=600)


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/api/health")
async def health():
    """Open http://localhost:8000/api/health to check both keys quickly."""
    out = {"model": agent.MODEL, "tavily_key_set": bool(agent.TAVILY_KEY)}
    try:
        r = await agent.llm.chat.completions.create(
            model=agent.MODEL, max_tokens=50, messages=[{"role": "user", "content": "Say hi"}])
        out["llm"] = "ok: " + (r.choices[0].message.content or "").strip()[:40]
    except Exception as e:
        out["llm"] = f"FAILED: {type(e).__name__}: {e}"
    return out


@app.post("/api/research")
async def run(body: Ask):
    async def stream():
        async with slots:
            try:
                async for ev in research(body.question.strip()):
                    if ev["type"] == "done":
                        rid = uuid.uuid4().hex[:10]
                        db.execute("INSERT INTO reports VALUES(?,?,?,?,?,?)",
                                   (rid, body.question, ev["report"], json.dumps(ev["sources"]),
                                    json.dumps({"seconds": ev["seconds"], "corrected": ev["corrected"]}), time.time()))
                        db.commit()
                        ev["id"] = rid
                    yield sse(ev)
            except openai.APIStatusError as e:
                traceback.print_exc()
                hint = " (rate limit: wait a minute or lower LLM_CONCURRENCY)" if e.status_code == 429 else ""
                yield sse({"type": "error", "text": f"LLM API error {e.status_code}: {e.message}{hint}"})
            except Exception as e:
                traceback.print_exc()
                yield sse({"type": "error", "text": f"{type(e).__name__}: {e}"})
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/reports")
def list_reports():
    rows = db.execute("SELECT id, question, created FROM reports ORDER BY created DESC LIMIT 30").fetchall()
    return [{"id": r[0], "question": r[1], "created": r[2]} for r in rows]


@app.get("/api/reports/{rid}")
def get_report(rid: str):
    r = db.execute("SELECT id, question, report, sources, meta FROM reports WHERE id=?", (rid,)).fetchone()
    if not r:
        raise HTTPException(404, "Report not found")
    return {"id": r[0], "question": r[1], "report": r[2], "sources": json.loads(r[3]), **json.loads(r[4])}


app.mount("/", StaticFiles(directory=Path(__file__).parent.parent / "frontend", html=True), name="ui")
