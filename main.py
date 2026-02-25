import os, json, base64, time, asyncio, logging, subprocess, sys
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
RATE_LIMIT = int(os.getenv("RATE_LIMIT", 20))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", 60))

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

K = (os.getenv("GROQ_API_KEY") or "").strip()
T = (os.getenv("GH_TOKEN") or "").strip()
R = (os.getenv("REPO_PATH") or "").strip()
STATE = {"rules": "Goal: AI Engineer. Strategy: Deep Reflection over Speed.", "lvl": 1}
CTX_MAX_CHARS = int(os.getenv("CTX_MAX_CHARS", 8000))

# ─── TOOLS ───────────────────────────────────────────────────────────────────

def fn_1_env(k="", **kwargs): return os.getenv(k, "Null")

def fn_2_log(m=None, **kwargs):
    msg = m or json.dumps(kwargs) or "Log recorded"
    logger.info(f"[Reflect]: {msg}")
    return "Log recorded"

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

JSON_ENFORCEMENT = (
    " Always respond with a single valid JSON object only — "
    "no markdown, no prose, no code fences. "
    'Schema: {"tool": "<n>", "args": {}, "thought": "<reasoning>"}.'
)

async def fn_commit(path, content, msg):
    try:
        if not T or not R: return "Save_Failed: Missing GH Config"
        headers = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}
        async with httpx.AsyncClient() as client:
            get_resp = await client.get(f"https://api.github.com/repos/{R}/contents/{path}", headers=headers)
            sha = get_resp.json().get("sha", "") if get_resp.status_code == 200 else ""
            put_resp = await client.put(
                f"https://api.github.com/repos/{R}/contents/{path}",
                headers=headers,
                json={"message": msg, "content": base64.b64encode(content.encode()).decode(), "sha": sha}
            )
            return f"Saved_{put_resp.status_code}"
    except Exception as e:
        logger.error(f"[Commit] Exception: {e}")
        return "Save_Failed"

# ─── LANGCHAIN & GITHUB TOOLS ────────────────────────────────────────────────

def _ensure_pkg(pkg, import_as=None):
    name = import_as or pkg.split("[")[0]
    try: __import__(name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

def fn_8_pip(package="", **kwargs):
    pkg = package or kwargs.get("pkg", "")
    try:
        _ensure_pkg(pkg)
        return f"Installed: {pkg}"
    except Exception as e: return f"pip_err: {e}"

def fn_9_lc_tool(tool="", input="", **kwargs):
    tool = tool or kwargs.get("name", "")
    inp  = input or kwargs.get("query", "")
    try:
        _ensure_pkg("langchain-community", "langchain_community")
        if tool in ("ddg-search", "search"):
            _ensure_pkg("duckduckgo-search", "duckduckgo_search")
            from langchain_community.tools import DuckDuckGoSearchRun
            return DuckDuckGoSearchRun().run(inp)[:1500]
        return f"lc_err: unknown tool '{tool}'"
    except Exception as e: return f"lc_err: {e}"

async def fn_read_github(path="", **kwargs):
    path = path or kwargs.get("file", "")
    if not T or not R: return "read_err: Missing Config"
    try:
        headers = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://api.github.com/repos/{R}/contents/{path}", headers=headers)
            if resp.status_code != 200: return f"read_err: {resp.status_code}"
            return base64.b64decode(resp.json()["content"]).decode("utf-8")[:6000]
    except Exception as e: return f"read_err: {e}"

async def fn_propose_patch(instruction="", **kwargs):
    instruction = instruction or kwargs.get("desc", "")
    current_code = await fn_read_github("main.py")
    agents_spec  = await fn_read_github("agents.md")
    prompt = (
        f"You are a Python engineer making a MINIMAL addition.\n"
        f"agents.md: {agents_spec[:600]}\n"
        f"main.py tail: {current_code[-800:]}\n"
        f"Instruction: {instruction}\n"
        "Output Raw Python only. Start with # --- PATCH: <desc> ---"
    )
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {K}"},
                json={
                    "model": "groq/compound",
                    "messages": [{"role": "system", "content": "Raw Python generator."}, {"role": "user", "content": prompt}],
                    "max_tokens": 1024
                }
            )
            proposed = resp.json()["choices"][0]["message"]["content"].strip()
            if proposed.startswith("```"): proposed = proposed.split("\n", 1)[1].rsplit("```", 1)[0]
            await fn_commit("main_patch.py", proposed, f"[AutoPatch] {instruction[:50]}")
            return "Patch saved to main_patch.py"
    except Exception as e: return f"patch_err: {e}"

async def fn_apply_patch(**kwargs):
    patch = await fn_read_github("main_patch.py")
    current = await fn_read_github("main.py")
    if "# --- PATCH:" not in patch: return "apply_err: invalid patch header"
    marker = "# ─── CATCH-ALL POST"
    updated = current.replace(marker, patch.strip() + "\n\n\n" + marker, 1) if marker in current else current + "\n\n" + patch
    return await fn_commit("main.py", updated, "[AutoApply] Patch application")

async def fn_align_with_spec(**kwargs):
    spec = await fn_read_github("agents.md")
    code = await fn_read_github("main.py")
    prompt = f"Identify the top missing capability in one sentence based on:\nSpec: {spec[:500]}\nCode: {code[:500]}"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://api.groq.com/openai/v1/chat/completions",
                                     headers={"Authorization": f"Bearer {K}"},
                                     json={"model": "groq/compound", "messages": [{"role": "user", "content": prompt}]})
            gap = resp.json()["choices"][0]["message"]["content"].strip()
            return await fn_propose_patch(instruction=gap)
    except Exception as e: return f"align_err: {e}"

TOOLS = {
    "env": fn_1_env, "log": fn_2_log, "math": fn_3_math, "fmt": fn_4_fmt,
    "chk": fn_5_chk, "ui": fn_6_ui, "mut": fn_7_mut, "commit": fn_commit,
    "pip": fn_8_pip, "lc": fn_9_lc_tool, "read": fn_read_github,
    "propose_patch": fn_propose_patch, "apply_patch": fn_apply_patch, "align": fn_align_with_spec
}

PRMPTS = [
    "Analyze state. Call tool='chk' with args={'g': '<gap>'}.",
    "Hypothesize. Call tool='log' with args={'m': '<hypothesis>'}.",
    "Identify failure. Call tool='fmt' with args={'d': '<failure>'}.",
    "Update rules. Call tool='mut' with args={'p': '<short_rules>'}.",
    "Summary. Call tool='log' with args={'m': '<summary>'}.",
    "MANDATORY: Call tool='align' with args={}."
]

# ─── GROQ RATE LIMITING & LLM CALL ───────────────────────────────────────────

GROQ_SEMAPHORE = asyncio.Semaphore(3)
GROQ_CALL_TIMES, GROQ_TOKEN_LOG, GROQ_DAY_CALLS = [], [], []
FALLBACK = '{"tool": "log", "args": {"m": "API Overload"}, "thought": "retry"}'

SYSTEM_PROMPT_TEMPLATE = (
    "{rules}. "
    "Respond with single valid JSON only. "
    "Schema: {{\"tool\": \"<name>\", \"args\": {{}}, \"thought\": \"<reasoning>\"}}. "
    "Tools: env, log, math, fmt, chk, ui, mut, pip, lc, read, propose_patch, apply_patch, align."
)

async def call_llm(p) -> str:
    async with GROQ_SEMAPHORE:
        await asyncio.sleep(1.5)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {K}"},
                    json={
                        "model": "groq/compound",
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(rules=STATE["rules"]) + JSON_ENFORCEMENT},
                            {"role": "user", "content": p}
                        ],
                        "response_format": {"type": "json_object"}
                    },
                    timeout=30.0
                )
                return response.json()["choices"][0]["message"]["content"]
        except Exception: return FALLBACK

# ─── AUTONOMOUS ENGINE ────────────────────────────────────────────────────────

async def run_autonomous_loop(input_str: str) -> str:
    ctx = input_str
    for i in range(6):
        payload = ctx[-CTX_MAX_CHARS:]
        raw = await call_llm(f"Context: {payload}. Directive: {PRMPTS[i]}")
        try:
            data = json.loads(raw)
        except:
            import re
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            data = json.loads(match.group()) if match else {"tool": "log", "args": {"m": "Parse Error"}}

        t, a = data.get("tool"), data.get("args", {})
        
        # Enforce step-specific tools
        if i == 3: t, a = "mut", {"p": a.get("p", "Goal: AI Engineer.")}
        if i == 4: t, a = "log", {"m": a.get("m", "Step 4 complete.")}
        if i == 5: t, a = "align", {}

        if t in TOOLS:
            res = TOOLS[t](**a)
            if asyncio.iscoroutine(res): res = await res
            ctx += f"\n[Step {i}] {t}: {res}"
    
    await fn_commit("engineer_log.md", ctx, "Intellectual Evolution Log")
    return ctx

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/status")
def status(): return {"status": "Active", "rules": STATE["rules"]}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    output = await run_autonomous_loop(body.get("input", ""))
    return {"ok": True, "output": output}

@app.api_route("/{full_path:path}", methods=["POST"])
async def catch_all_post(full_path: str, request: Request):
    body = await request.json()
    trigger = body.get("input") or f"POST /{full_path}"
    asyncio.create_task(run_autonomous_loop(trigger))
    return {"status": "Agent loop started", "trigger": trigger}
