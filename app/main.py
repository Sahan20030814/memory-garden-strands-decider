"""Memory Garden: Strands Decider 2B gates which memories reach the Bedrock LLM."""
import json, os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import boto3, httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DECIDER = os.getenv("DECIDER_URL", "http://127.0.0.1:8099")
MODEL = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
REGION = os.getenv("AWS_REGION", "us-east-1")

# ---- the reusable decision pattern: 4 typed yes/no questions per memory chunk
def yn(q, t, f):
    return {"type": "noul", "instructions": q, "criteria": {"true": t, "false": f}}
QUESTIONS = {
    "relevant": yn("Does this memory help answer the user's message?", "it holds facts needed for this message", "it is unrelated to this message"),
    "stale": yn("Is this memory outdated, no longer true today?", "it describes a past situation that has changed", "it is still current"),
    "superseded": yn("Is this memory contradicted by one of the newer memories listed?", "a newer memory replaces or contradicts it", "no newer memory conflicts with it"),
    "private": yn("Is this memory private (health, money, credentials) and unnecessary for this message?", "sensitive detail the message does not need", "not sensitive, or needed"),
}

MEM = [dict(id=i, topic=t, age=a, text=x) for i, (t, a, x) in enumerate([
    ("job", 400, "Worked as a junior developer at a software company in Kandy."),
    ("job", 20, "Now a backend engineer at an AWS partner company."),
    ("diet", 200, "Is vegetarian."),
    ("diet", 30, "Eats fish again, no longer vegetarian."),
    ("pet", 90, "Has a dog named Biscuit."),
    ("health", 60, "Takes medication for high blood pressure."),
    ("money", 45, "Owes 40,000 LKR on a credit card."),
    ("travel", 15, "Plans a train trip to Ella next month."),
    ("lang", 300, "Prefers replies in a Sinhala-English mix."),
    ("tools", 120, "Uses VS Code, Python and Docker every day."),
    ("goal", 10, "Wants to win the AWS Agent Experience Hackathon."),
    ("music", 150, "Likes lo-fi and old Sinhala songs."),
    ("city", 250, "Lives in Kandy."),
    ("city", 25, "Moved to Colombo for the new job."),
    ("learn", 70, "Learns Rust on weekends."),
    ("drink", 80, "Drinks tea, never coffee."),
])]

def state_for(q, m):
    s = f"User message: {q}\nMemory (saved {m['age']} days ago, topic {m['topic']}): {m['text']}"
    newer = [x for x in MEM if x["topic"] == m["topic"] and x["age"] < m["age"]]
    if newer:
        s += "\nNewer memories on this topic:\n" + "\n".join(f"- ({x['age']} days ago) {x['text']}" for x in newer)
    return s

def dig(o, k):  # tolerant reader: finds question k anywhere in the response
    if isinstance(o, dict):
        if k in o:
            v = o[k]
            if isinstance(v, dict):
                v = v.get("noul", v.get("p_true", v.get("probability")))
            return float(v)
        for v in o.values():
            r = dig(v, k)
            if r is not None: return r
    elif isinstance(o, list):
        for v in o:
            r = dig(v, k)
            if r is not None: return r

def fallback(q, m):  # clearly labelled keyword heuristic, only when the Decider server is down
    w = lambda s: set(re.findall(r"[a-z]{4,}", s.lower()))
    rel = min(1, len(w(q) & (w(m["text"]) | {m["topic"]})) / 2)
    return dict(relevant=rel, stale=min(1, m["age"] / 400), superseded=0.0, private=1.0 if m["topic"] in ("health", "money") else 0.0)

def score(q, m):
    t = time.time()
    try:
        r = httpx.post(f"{DECIDER}/v1/systemone", json={"state": state_for(q, m), "questions": QUESTIONS}, timeout=60).json()
        p = {k: dig(r, k) for k in QUESTIONS}
        if None in p.values(): raise ValueError(f"unexpected response shape: {r}")
        mode = "decider"
    except httpx.HTTPError:
        p, mode = fallback(q, m), "fallback"
    return p, mode, int((time.time() - t) * 1000)

def verdict(p, th):
    if p["private"] > .6 and p["relevant"] < .8: return "withheld"
    if p["superseded"] > .6 or p["stale"] > .75: return "pruned"
    return "injected" if p["relevant"] >= th else "skipped"

app = FastAPI()
class Ask(BaseModel):
    query: str
    threshold: float = 0.5
def sse(e, d): return f"event: {e}\ndata: {json.dumps(d)}\n\n"

@app.get("/api/memories")
def memories(): return MEM

@app.post("/api/ask")
def ask(req: Ask):
    def gen():
        t0, res = time.time(), {}
        with ThreadPoolExecutor(4) as ex:
            futs = {ex.submit(score, req.query, m): m for m in MEM}
            for f in as_completed(futs):
                p, mode, ms = f.result()
                res[futs[f]["id"]] = d = dict(id=futs[f]["id"], p=p, v=verdict(p, req.threshold), ms=ms, mode=mode)
                yield sse("chunk", d)
        keep = [m for m in MEM if res[m["id"]]["v"] == "injected"]
        tok = lambda ms: sum(len(m["text"]) for m in ms) // 4
        yield sse("stats", dict(gate_ms=int((time.time() - t0) * 1000), kept=len(keep), total=len(MEM), tokens_all=tok(MEM), tokens_kept=tok(keep)))
        system = "You are a warm, concise assistant. Use only these memories about the user, and only when relevant:\n" + "\n".join("- " + m["text"] for m in keep)
        if os.getenv("MOCK_LLM"):  # demo mode: no Bedrock call
            text = "Demo mode, no Bedrock call. I would answer using only: " + "; ".join(m["text"] for m in keep) if keep else "Demo mode. No memory looked relevant, so I would answer without any."
            for w in text.split(" "):
                time.sleep(0.03)
                yield sse("token", {"t": w + " "})
            yield sse("done", {})
            return
        try:
            br = boto3.client("bedrock-runtime", region_name=REGION)
            r = br.converse_stream(modelId=MODEL, system=[{"text": system}], messages=[{"role": "user", "content": [{"text": req.query}]}])
            for ev in r["stream"]:
                t = ev.get("contentBlockDelta", {}).get("delta", {}).get("text")
                if t: yield sse("token", {"t": t})
        except Exception as e:
            yield sse("error", {"msg": f"Bedrock call failed: {e}"})
        yield sse("done", {})
    return StreamingResponse(gen(), media_type="text/event-stream")

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "web")), name="static")
@app.get("/")
def index(): return FileResponse(os.path.join(os.path.dirname(__file__), "..", "web", "index.html"))
