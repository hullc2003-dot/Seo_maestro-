import os, json, base64, time, asyncio, logging
import httpx
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

# ─── CONTEXT TRUNCATION LIMIT (chars) — prevents Groq 413 errors ─────────────

CTX_MAX_CHARS = int(os.getenv("CTX_MAX_CHARS", 8000))

# ─── TOOLS ───────────────────────────────────────────────────────────────────

def fn_1_env(k="", **kwargs): return os.getenv(k, "Null")

# FIX 1: default m=None and **kwargs absorb unknown keyword args the LLM may send

def fn_2_log(m=None, **kwargs):
    msg = m or json.dumps(kwargs) or "Log recorded"
    logger.info(f"[Reflect]: {msg}")
    return "Log recorded"

# FIX 3: replaced eval with simpleeval to prevent code execution via user input

def fn_3_math(e):
    try:
        from simpleeval import simple_eval
        return simple_eval(e)
    except Exception:
        return "Math Err"

def fn_4_fmt(d="", **kwargs): return f"### ANALYSIS ###\n{d or json.dumps(kwargs)}"
def fn_5_chk(g="", **kwargs): return f"Goal Alignment: {g or json.dumps(kwargs)}"
def fn_6_ui(d="", **kwargs): return f"UI_UPDATE: {d or json.dumps(kwargs)}"
def fn_7_mut(p="", **kwargs):
    new_rules = p or kwargs.get("rules") or kwargs.get("ruleset") or ""
    if not new_rules or not new_rules.strip():
        logger.warning("[mut] Rejected empty ruleset — STATE['rules'] unchanged")
        return "Mut rejected: empty ruleset"
    STATE["rules"] = new_rules.strip()
    return "Core Rules Redefined"

# FIX 2: replaced blocking http.client with async httpx

async def fn_commit(path, content, msg):
    try:
        if not T:
            logger.error("[Commit] GH_TOKEN env var is not set")
            return "Save_Failed: no GH_TOKEN"
        if not R:
            logger.error("[Commit] REPO_PATH env var is not set")
            return "Save_Failed: no REPO_PATH"
        headers = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}
        async with httpx.AsyncClient() as client:
            get_resp = await client.get(
                f"https://api.github.com/repos/{R}/contents/{path}",
                headers=headers
            )
            get_data = get_resp.json()
            if get_resp.status_code not in (200, 404):
                logger.error(f"[Commit] GET failed {get_resp.status_code}: {get_data}")
                return f"Save_Failed: GET {get_resp.status_code}"
            sha = get_data.get("sha", "")
            put_resp = await client.put(
                f"https://api.github.com/repos/{R}/contents/{path}",
                headers=headers,
                json={"message": msg, "content": base64.b64encode(content.encode()).decode(), "sha": sha}
            )
            put_data = put_resp.json()
            if put_resp.status_code not in (200, 201):
                logger.error(f"[Commit] PUT failed {put_resp.status_code}: {put_data}")
                return f"Save_Failed: PUT {put_resp.status_code}"
            return f"Saved_{put_resp.status_code}"
    except Exception as e:
        logger.error(f"[Commit] Exception: {e}", exc_info=True)
        return "Save_Failed"

TOOLS = {
    "env": fn_1_env, "log": fn_2_log, "math": fn_3_math,
    "fmt": fn_4_fmt, "chk": fn_5_chk, "ui": fn_6_ui,
    "mut": fn_7_mut, "commit": fn_commit
}

PRMPTS = [
    "Critically analyze the current state. What is missing to reach AI Engineer status? You MUST call tool='chk' with args={'g': '<your one-sentence gap summary>'}.",
    "Generate a hypothesis for a better autonomous pattern. You MUST call tool='log' with args={'m': '<your hypothesis in one sentence>'}.",
    "Identify the single most critical failure point in the previous step. You MUST call tool='fmt' with args={'d': '<failure point in one sentence>'}.",
    "Rewrite the current ruleset to prevent premature task exit. You MUST call tool='mut' with args={'p': '<your new ruleset as a single string>'}.",
    "Log a one-sentence final verification summary. You MUST call tool='log' with args={'m': '<verification summary>'}.",
]

# ─── GROQ RATE LIMITING ───────────────────────────────────────────────────────

GROQ_SEMAPHORE = asyncio.Semaphore(3)

GROQ_CALL_TIMES: list[float] = []
GROQ_TOKEN_LOG: list[tuple[float, int]] = []   # (timestamp, tokens_used)
GROQ_DAY_CALLS: list[float] = []               # timestamps for RPD tracking

GROQ_RPM_LIMIT  = int(os.getenv("GROQ_RPM_LIMIT",  25))      # requests/min
GROQ_TPM_LIMIT  = int(os.getenv("GROQ_TPM_LIMIT", 28000))  # tokens/min
GROQ_RPD_LIMIT  = int(os.getenv("GROQ_RPD_LIMIT",  250))     # requests/day

# ─── LLM CALL ─────────────────────────────────────────────────────────────────

FALLBACK = '{"tool": "log", "args": {"m": "API Overload"}, "thought": "retry"}'

# FIX 3: Strengthened system prompt to enforce JSON-only output and document
# the exact allowed tool names + arg schema so the LLM doesn't invent fields.

SYSTEM_PROMPT_TEMPLATE = (
    "{rules}. "
    "You MUST respond with a single valid JSON object and nothing else — "
    "no markdown, no prose, no code fences. "
    "Your entire response must be parseable by json.loads(). "
    "Schema: {{\"tool\": \"<name>\", \"args\": {{}}, \"thought\": \"<reasoning>\"}}. "
    "Valid tools: env(k), log(m), math(e), fmt(d), chk(g), ui(d), mut(p), commit(path,content,msg). "
    "Use exactly these argument names. Do not add extra fields."
)

async def call_llm(p) -> str:
    async with GROQ_SEMAPHORE:
        now = time.time()

        # ── Sliding windows ──────────────────────────────────────────────────
        GROQ_CALL_TIMES[:] = [t for t in GROQ_CALL_TIMES if now - t < 60]
        GROQ_TOKEN_LOG[:]  = [(t, tk) for t, tk in GROQ_TOKEN_LOG if now - t < 60]
        GROQ_DAY_CALLS[:]  = [t for t in GROQ_DAY_CALLS if now - t < 86_400]

        # ── RPD guard ────────────────────────────────────────────────────────
        if len(GROQ_DAY_CALLS) >= GROQ_RPD_LIMIT:
            wait = 86_400 - (now - GROQ_DAY_CALLS[0])
            logger.warning(f"[Groq throttle] RPD cap hit — waiting {wait:.0f}s (~{wait/3600:.1f}h)")
            await asyncio.sleep(wait)

        # ── RPM guard ────────────────────────────────────────────────────────
        if len(GROQ_CALL_TIMES) >= GROQ_RPM_LIMIT:
            wait = 60 - (now - GROQ_CALL_TIMES[0])
            logger.warning(f"[Groq throttle] RPM cap hit — waiting {wait:.1f}s")
            await asyncio.sleep(wait)

        # ── TPM guard ────────────────────────────────────────────────────────
        tokens_used = sum(tk for _, tk in GROQ_TOKEN_LOG)
        if tokens_used >= GROQ_TPM_LIMIT:
            wait = 60 - (now - GROQ_TOKEN_LOG[0][0])
            logger.warning(f"[Groq throttle] TPM cap hit ({tokens_used} used) — waiting {wait:.1f}s")
            await asyncio.sleep(wait)

        # ── Register this call ───────────────────────────────────────────────
        ts = time.time()
        GROQ_CALL_TIMES.append(ts)
        GROQ_DAY_CALLS.append(ts)
        await asyncio.sleep(1.5)

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {K}", "Content-Type": "application/json"},
                    json={
                        "model": "groq/compound",
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(rules=STATE["rules"])},
                            {"role": "user", "content": p}
                        ],
                        "response_format": {"type": "json_object"}
                    },
                    timeout=30.0
                )
                resp = response.json()

            if "choices" not in resp:
                err = resp.get("error", {})
                if err.get("type") == "permissions_error":
                    logger.error(f"[Groq] Permissions error — aborting: {err.get('message')}")
                    raise RuntimeError(f"Groq permissions error: {err.get('message')}")
                logger.error(f"[Groq] Unexpected response (no 'choices'): {resp}")
                GROQ_TOKEN_LOG.append((time.time(), 0))
                return FALLBACK

            usage = resp.get("usage", {})
            total_tokens = usage.get("total_tokens", 500)
            GROQ_TOKEN_LOG.append((time.time(), total_tokens))
            logger.info(f"[Groq] tokens this call: {total_tokens} | TPM window: {sum(tk for _, tk in GROQ_TOKEN_LOG)} | RPD: {len(GROQ_DAY_CALLS)}/250")

            return resp["choices"][0]["message"]["content"]

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"[Groq] API call failed: {e}", exc_info=True)
            GROQ_TOKEN_LOG.append((time.time(), 500))
            return FALLBACK

# ─── AUTONOMOUS ENGINE ────────────────────────────────────────────────────────

async def run_autonomous_loop(input_str: str) -> str:
    ctx = input_str
    for i in range(5):
        # FIX 4: truncate ctx before sending to prevent Groq 413 errors
        ctx_payload = ctx[-CTX_MAX_CHARS:] if len(ctx) > CTX_MAX_CHARS else ctx

        raw = await call_llm(f"PRE-STEP REFLECTION. Current Context: {ctx_payload}. Directive: {PRMPTS[i % 5]}")

        if not raw:
            logger.warning(f"[Loop] Step {i}: call_llm returned empty, skipping")
            continue

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"[Loop] Step {i}: failed to parse LLM response: {e} | raw={raw!r}")
            continue

        t, a = data.get("tool"), data.get("args", {})

        # FIX 5: skip cleanly when LLM emits tool="none" or omits tool entirely
        if not t or t.lower() == "none":
            logger.info(f"[Loop] Step {i}: LLM chose no tool — skipping")
            ctx += f"\n[Step {i}] No action taken | Reasoning: {data.get('thought')}"
            continue

        # FIX: prevent LLM from invoking commit mid-loop; it runs once after all steps
        if t == "commit":
            logger.warning(f"[Loop] Step {i}: LLM tried to call 'commit' mid-loop — blocked")
            ctx += f"\n[Step {i}] Commit blocked (runs at end) | Reasoning: {data.get('thought')}"
            continue

        if t in TOOLS:
            try:
                res = TOOLS[t](**a)
                if asyncio.iscoroutine(res):
                    res = await res
            except Exception as e:
                res = f"Tool error: {e}"
                logger.error(f"[Loop] Step {i}: tool '{t}' raised: {e}", exc_info=True)
            ctx += f"\n[Step {i}] Action: {t} | Result: {res} | Reasoning: {data.get('thought')}"
        else:
            logger.warning(f"[Loop] Step {i}: unknown or missing tool '{t}'")

    commit_result = await fn_commit("engineer_log.md", ctx, "Intellectual Evolution Log")
    logger.info(f"[Commit] {commit_result}")
    return ctx

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def status():
    return {"status": "Deep Thinking", "rules": STATE["rules"], "lvl": STATE["lvl"]}

# ─── CHAT ROUTE ───────────────────────────────────────────────────────────────

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
