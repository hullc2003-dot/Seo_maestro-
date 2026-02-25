import os, json, http.client, base64, time, asyncio, logging
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AgentServer")

# ─── MIDDLEWARE: Request Logger ───────────────────────────────────────────────

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        logger.info(f"→ {request.method} {request.url.path} | Client: {request.client.host}")
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"← {response.status_code} | {elapsed:.1f}ms")
        return response

# ─── MIDDLEWARE: Rate Limiter (per IP, in-memory) ─────────────────────────────

RATE_STORE: dict[str, list[float]] = {}
RATE_LIMIT = int(os.getenv("RATE_LIMIT", 20))   # requests
RATE_WINDOW = int(os.getenv("RATE_WINDOW", 60))  # seconds

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host
        now = time.time()
        hits = [t for t in RATE_STORE.get(ip, []) if now - t < RATE_WINDOW]
        if len(hits) >= RATE_LIMIT:
            return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
        hits.append(now)
        RATE_STORE[ip] = hits
        return await call_next(request)

# ─── MIDDLEWARE: Error Handler ────────────────────────────────────────────────

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse({"error": "Internal server error", "detail": str(exc)}, status_code=500)

app = FastAPI(title="Autonomous Agent", version="2.0")

# ─── REGISTER MIDDLEWARES (order matters: first added = outermost) ────────────

app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(","),
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

K = os.getenv("GROQ_API_KEY")
T = os.getenv("GH_TOKEN")
R = os.getenv("REPO_PATH")
STATE = {"rules": "Goal: AI Engineer. Strategy: Deep Reflection over Speed.", "lvl": 1}

# ─── TOOLS ───────────────────────────────────────────────────────────────────

def fn_1_env(k): return os.getenv(k, "Null")
def fn_2_log(m): logger.info(f"[Reflect]: {m}"); return "Log recorded"
def fn_3_math(e):
    try: return eval(e, {"__builtins__": None}, {})
    except: return "Math Err"
def fn_4_fmt(d): return f"### ANALYSIS ###\n{d}"
def fn_5_chk(g): return f"Goal Alignment: {g}"
def fn_6_ui(d): return f"UI_UPDATE: {d}"
def fn_7_mut(p): STATE["rules"] = p; return "Core Rules Redefined"

def fn_commit(path, content, msg):
    try:
        c = http.client.HTTPSConnection("api.github.com")
        h = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}
        c.request("GET", f"/repos/{R}/contents/{path}", headers=h)
        sha = json.loads(c.getresponse().read()).get("sha", "")
        d = json.dumps({"message": msg, "content": base64.b64encode(content.encode()).decode(), "sha": sha})
        c.request("PUT", f"/repos/{R}/contents/{path}", d, h)
        return f"Saved_{c.getresponse().status}"
    except: return "Save_Failed"

TOOLS = {
    "env": fn_1_env, "log": fn_2_log, "math": fn_3_math,
    "fmt": fn_4_fmt, "chk": fn_5_chk, "ui": fn_6_ui,
    "mut": fn_7_mut, "commit": fn_commit
}

PRMPTS = [
    "Critically analyze the current state. What is missing to reach AI Engineer status?",
    "Generate a hypothesis for a better autonomous pattern. Test it via 'log'.",
    "Identify potential failure points in the previous step's output.",
    "Refine the current ruleset using 'mut' to prevent premature task exit.",
    "Execute a final verification and commit the refined knowledge base."
]

# ─── GROQ RATE LIMITING ───────────────────────────────────────────────────────

GROQ_SEMAPHORE = asyncio.Semaphore(3)   # max 3 concurrent LLM calls
GROQ_CALL_TIMES: list[float] = []
GROQ_RPM_LIMIT = int(os.getenv("GROQ_RPM_LIMIT", 25))  # stay under Groq's 30 RPM free tier cap

# ─── LLM CALL ─────────────────────────────────────────────────────────────────

async def call_llm(p):
    async with GROQ_SEMAPHORE:
        # Sliding window — drop timestamps older than 60s
        now = time.time()
        GROQ_CALL_TIMES[:] = [t for t in GROQ_CALL_TIMES if now - t < 60]

        if len(GROQ_CALL_TIMES) >= GROQ_RPM_LIMIT:
            wait = 60 - (now - GROQ_CALL_TIMES[0])
            logger.warning(f"[Groq throttle] RPM cap hit — waiting {wait:.1f}s")
            await asyncio.sleep(wait)

        GROQ_CALL_TIMES.append(time.time())
        await asyncio.sleep(1.5)  # base inter-call spacing

        try:
            c = http.client.HTTPSConnection("api.groq.com")
            body = json.dumps({
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": f"{STATE['rules']}. Response MUST be JSON: {{'tool': 'name', 'args': {{}}, 'thought': 'reasoning'}}"},
                    {"role": "user", "content": p}
                ],
                "response_format": {"type": "json_object"}
            })
            c.request("POST", "/openai/v1/chat/completions", body,
                      {"Authorization": f"Bearer {K}", "Content-Type": "application/json"})
            return json.loads(c.getresponse().read().decode())["choices"][0]["message"]["content"]
        except:
            return '{"tool": "log", "args": {"m": "API Overload"}, "thought": "retry"}'

# ─── AUTONOMOUS ENGINE ────────────────────────────────────────────────────────

async def run_autonomous_loop(input_str: str):
    ctx = input_str
    for i in range(5):
        raw = await call_llm(f"PRE-STEP REFLECTION. Current Context: {ctx}. Directive: {PRMPTS[i % 5]}")
        data = json.loads(raw)
        t, a = data.get("tool"), data.get("args", {})
        if t in TOOLS:
            res = TOOLS[t](**a)
            ctx += f"\n[Step {i}] Action: {t} | Result: {res} | Reasoning: {data.get('thought')}"
        fn_commit("engineer_log.md", ctx, "Intellectual Evolution Log")
    return ctx

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    return {"status": "Deep Thinking", "rules": STATE["rules"], "lvl": STATE["lvl"]}

# ─── CHAT ROUTE (matches your UI exactly) ─────────────────────────────────────

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        trigger = body.get("input", "").strip()
        if not trigger:
            return {"ok": False, "error": "No input provided"}
        output = await run_autonomous_loop(trigger)
        return {"ok": True, "output": output}
    except Exception as e:
        logger.error(f"[Chat] Error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}

@app.post("/deploy")
async def deploy(request: Request):
    body = await request.json()
    trigger = body.get("input", "Manual Trigger via /deploy")
    asyncio.create_task(run_autonomous_loop(trigger))
    return {"status": "Agent loop started", "trigger": trigger}

# ─── CATCH-ALL POST ───────────────────────────────────────────────────────────

@app.api_route("/{full_path:path}", methods=["POST"])
async def catch_all_post(full_path: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    trigger = (
        body.get("input")
        or body.get("message")
        or body.get("text")
        or body.get("prompt")
        or (json.dumps(body) if body else f"POST /{full_path}")
    )

    logger.info(f"[Catch-All] /{full_path} → trigger: {trigger[:80]}")
    asyncio.create_task(run_autonomous_loop(trigger))
    return {"status": "Agent loop started", "path": f"/{full_path}", "trigger": trigger}

