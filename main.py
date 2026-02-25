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

K = (os.getenv("GROQ_API_KEY") or "").strip()
T = (os.getenv("GH_TOKEN") or "").strip()
R = (os.getenv("REPO_PATH") or "").strip()
STATE = {"rules": "Goal: AI Engineer. Strategy: Deep Reflection over Speed.", "lvl": 1}

# ─── CONTEXT TRUNCATION LIMIT (chars) — prevents Groq 413 errors ─────────────

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
            saved = f"Saved_{put_resp.status_code}"
            asyncio.ensure_future(signal_ui(f"Committed {path}"))
            return saved
    except Exception as e:
        logger.error(f"[Commit] Exception: {e}", exc_info=True)
        return "Save_Failed"

# ─── LANGCHAIN BOOTSTRAP ─────────────────────────────────────────────────────

def _ensure_pkg(pkg, import_as=None):
    """Install a package at runtime if missing."""
    name = import_as or pkg.split("[")[0]
    try:
        __import__(name)
    except ImportError:
        logger.info(f"[Bootstrap] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

def fn_8_pip(package="", **kwargs):
    """Install a Python package so it can be used in future tool calls."""
    pkg = package or kwargs.get("pkg", "")
    if not pkg:
        return "pip_err: no package name"
    try:
        _ensure_pkg(pkg)
        return f"Installed: {pkg}"
    except Exception as e:
        logger.error(f"[pip] {e}")
        return f"pip_err: {e}"

def fn_9_lc_tool(tool="", input="", **kwargs):
    """Run a LangChain community tool by name (e.g. 'ddg-search', 'wikipedia')."""
    tool = tool or kwargs.get("name", "")
    inp  = input or kwargs.get("query", "") or kwargs.get("input", "")
    if not tool or not inp:
        return "lc_err: need tool and input"
    try:
        _ensure_pkg("langchain-community", "langchain_community")
        _ensure_pkg("duckduckgo-search",   "duckduckgo_search")
        if tool in ("ddg-search", "search", "web"):
            from langchain_community.tools import DuckDuckGoSearchRun
            return DuckDuckGoSearchRun().run(inp)[:1500]
        if tool in ("wikipedia", "wiki"):
            _ensure_pkg("wikipedia", "wikipedia")
            from langchain_community.tools import WikipediaQueryRun
            from langchain_community.utilities import WikipediaAPIWrapper
            return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run(inp)[:1500]
        return f"lc_err: unknown tool '{tool}'"
    except Exception as e:
        logger.error(f"[lc_tool] {e}", exc_info=True)
        return f"lc_err: {e}"

# ─── GITHUB FILE READER ───────────────────────────────────────────────────────

async def fn_read_github(path="", **kwargs):
    """Read any file from the repo. Use path='agents.md' or path='main.py'."""
    path = path or kwargs.get("file", "")
    if not path:
        return "read_err: no path"
    if not T or not R:
        return "read_err: missing GH_TOKEN or REPO_PATH"
    try:
        headers = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.github.com/repos/{R}/contents/{path}",
                headers=headers, timeout=15.0
            )
            data = resp.json()
            if resp.status_code != 200:
                return f"read_err: {resp.status_code} {data.get('message','')}"
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            logger.info(f"[ReadGitHub] {path} ({len(content)} chars)")
            return content[:6000]
    except Exception as e:
        logger.error(f"[ReadGitHub] {e}", exc_info=True)
        return f"read_err: {e}"

# ─── SELF-PATCH TOOLS ─────────────────────────────────────────────────────────

async def fn_propose_patch(instruction="", **kwargs):
    """Ask the LLM to generate a targeted code patch aligning main.py with agents.md."""
    instruction = instruction or kwargs.get("desc", "")
    if not instruction:
        return "patch_err: no instruction"
    current_code = await fn_read_github("main.py")
    agents_spec  = await fn_read_github("agents.md")
    if "read_err" in current_code:
        return f"patch_err: could not read main.py — {current_code}"
    if "read_err" in agents_spec:
        return f"patch_err: could not read agents.md — {agents_spec}"

    prompt = (
        f"You are a Python engineer making a MINIMAL addition to an existing file.\n\n"
        f"agents.md spec (excerpt):\n{agents_spec[:600]}\n\n"
        f"Current main.py tail (last 800 chars):\n{current_code[-800:]}\n\n"
        f"Instruction: {instruction}\n\n"
        "Write ONLY the new Python code to ADD (functions, constants, or route handlers). "
        "Do NOT rewrite the whole file. Do NOT include imports that already exist. "
        "Output must start with a comment like # --- PATCH: <description> ---\n"
        "No markdown fences. Raw Python only. Max 60 lines."
    )
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {K}", "Content-Type": "application/json"},
                json={
                    "model": "groq/compound",
                    "messages": [
                        {"role": "system", "content": "You are a Python code generator. Output raw Python only."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 4096
                },
                timeout=60.0
            )
        rdata2 = resp.json()
        if "choices" not in rdata2:
            err = rdata2.get("error", rdata2)
            logger.error(f"[ProposePatch] Groq call failed: {err}")
            return f"patch_err: groq={err}"
        proposed = rdata2["choices"][0]["message"]["content"].strip()
        if proposed.startswith("```"):
            proposed = proposed.split("\n", 1)[1].rsplit("```", 1)[0]
        result = await fn_commit("main_patch.py", proposed, f"[AutoPatch] {instruction[:80]}")
        return f"Proposed patch committed as main_proposed.py — {result}"
    except Exception as e:
        logger.error(f"[ProposePatch] {e}", exc_info=True)
        return f"patch_err: {e}"

async def fn_apply_patch(**kwargs):
    """Append main_patch.py content into main.py on GitHub, then redeploy."""
    patch = await fn_read_github("main_patch.py")
    if "read_err" in patch:
        return f"apply_err: {patch}"
    if "# --- PATCH:" not in patch:
        return "apply_err: patch missing header comment — unsafe to apply"
    current = await fn_read_github("main.py")
    if "read_err" in current:
        return f"apply_err: could not read main.py — {current}"
    
    insert_marker = "# --- PATCH: Contact UI signaling ---"
    
    if insert_marker in current:
        updated = current.replace(insert_marker, patch.strip() + "\n\n\n" + insert_marker, 1)
    else:
        updated = current + "\n\n" + patch.strip()
    
    if "FastAPI" not in updated or "async def run_autonomous_loop" not in updated:
        return "apply_err: merged file failed sanity check — autonomous loop missing"
    
    result = await fn_commit("main.py", updated, f"[AutoApply] Append patch to main.py")
    return f"main.py patched — {result}"

# ─── AGENTS.MD ALIGNMENT STEP ────────────────────────────────────────────────

async def fn_align_with_spec(**kwargs):
    agents_spec = await fn_read_github("agents.md")
    if "read_err" in agents_spec:
        return f"align_skipped: {agents_spec}"
    current_code = await fn_read_github("main.py")
    gap_prompt = (
        f"agents.md specification:\n{agents_spec[:800]}\n\n"
        f"Current main.py (truncated):\n{current_code[:600]}\n\n"
        "In one sentence, what is the single most important capability in agents.md "
        "that is missing or incomplete in main.py right now?"
    )
    try:
        await asyncio.sleep(8)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {K}", "Content-Type": "application/json"},
                json={
                    "model": "groq/compound",
                    "messages": [{"role": "user", "content": gap_prompt}],
                    "max_tokens": 200
                },
                timeout=30.0
            )
        rdata = resp.json()
        if "choices" not in rdata:
            err = rdata.get("error", rdata)
            fn_2_log(m=f"[Align] agents.md exists but gap LLM call failed")
            return f"align_partial: {err}"
        gap = rdata["choices"][0]["message"]["content"].strip()
        await asyncio.sleep(10)
        patch_result = await fn_propose_patch(instruction=gap)
        return f"Gap: {gap} | {patch_result}"
    except Exception as e:
        logger.error(f"[Align] {e}", exc_info=True)
        return f"align_err: {e}"

TOOLS = {
    "env": fn_1_env, "log": fn_2_log, "math": fn_3_math,
    "fmt": fn_4_fmt, "chk": fn_5_chk, "ui": fn_6_ui,
    "mut": fn_7_mut, "commit": fn_commit,
    "pip": fn_8_pip, "lc": fn_9_lc_tool,
    "read": fn_read_github, "propose_patch": fn_propose_patch,
    "apply_patch": fn_apply_patch, "align": fn_align_with_spec
}

PRMPTS = [
    "Critically analyze the current state. What is missing to reach AI Engineer status? You MUST call tool='chk' with args={'g': '<your one-sentence gap summary>'}.",
    "Generate a hypothesis for a better autonomous pattern. You MUST call tool='log' with args={'m': '<your hypothesis in one sentence>'}.",
    "Identify the single most critical failure point in the previous step. You MUST call tool='fmt' with args={'d': '<failure point in one sentence>'}.",
    "You MUST call tool='mut' with args={'p': '<one sentence ruleset>'}. Keep p under 100 characters. No markdown.",
    "Log a one-sentence final verification summary. You MUST call tool='log' with args={'m': '<verification summary>'}.",
    "MANDATORY FINAL STEP — no other tool is valid here. You MUST call tool='align' with args={}. Do NOT call chk, log, or any other tool."
]

# ─── GROQ RATE LIMITING ───────────────────────────────────────────────────────

GROQ_SEMAPHORE = asyncio.Semaphore(3)
GROQ_CALL_TIMES = []
GROQ_TOKEN_LOG = []
GROQ_DAY_CALLS = []

GROQ_RPM_LIMIT  = int(os.getenv("GROQ_RPM_LIMIT",  25))
GROQ_TPM_LIMIT  = int(os.getenv("GROQ_TPM_LIMIT", 28000))
GROQ_RPD_LIMIT  = int(os.getenv("GROQ_RPD_LIMIT",  250))

# ─── LLM CALL ─────────────────────────────────────────────────────────────────

FALLBACK = '{"tool": "log", "args": {"m": "API Overload"}, "thought": "retry"}'

SYSTEM_PROMPT_TEMPLATE = (
    "{rules}. "
    "You MUST respond with a single valid JSON object and nothing else — "
    "no markdown, no prose, no code fences. "
    "Your entire response must be parseable by json.loads(). "
    'Schema: {"tool": "<name>", "args": {}, "thought": "<reasoning>"}. '
    "Valid tools: env(k), log(m), math(e), fmt(d), chk(g), ui(d), mut(p), pip(package), lc(tool,input), read(path), propose_patch(instruction), apply_patch(), align(). "
    "Use exactly these argument names. Do not add extra fields."
)

async def call_llm(p) -> str:
    async with GROQ_SEMAPHORE:
        now = time.time()
        GROQ_CALL_TIMES[:] = [t for t in GROQ_CALL_TIMES if now - t < 60]
        GROQ_TOKEN_LOG[:]  = [(t, tk) for t, tk in GROQ_TOKEN_LOG if now - t < 60]
        GROQ_DAY_CALLS[:]  = [t for t in GROQ_DAY_CALLS if now - t < 86400]

        if len(GROQ_DAY_CALLS) >= GROQ_RPD_LIMIT:
            wait = 86400 - (now - GROQ_DAY_CALLS[0])
            await asyncio.sleep(wait)

        if len(GROQ_CALL_TIMES) >= GROQ_RPM_LIMIT:
            wait = 60 - (now - GROQ_CALL_TIMES[0])
            await asyncio.sleep(wait)

        tokens_used = sum(tk for _, tk in GROQ_TOKEN_LOG)
        if tokens_used >= GROQ_TPM_LIMIT:
            wait = 60 - (now - GROQ_TOKEN_LOG[0][0])
            await asyncio.sleep(wait)

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
                            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(rules=STATE["rules"]) + JSON_ENFORCEMENT},
                            {"role": "user", "content": p}
                        ],
                        "response_format": {"type": "json_object"}
                    },
                    timeout=30.0
                )
                resp = response.json()

            if "choices" not in resp:
                GROQ_TOKEN_LOG.append((time.time(), 0))
                return FALLBACK

            usage = resp.get("usage", {})
            total_tokens = usage.get("total_tokens", 500)
            GROQ_TOKEN_LOG.append((time.time(), total_tokens))
            return resp["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"[Groq] API call failed: {e}")
            GROQ_TOKEN_LOG.append((time.time(), 500))
            return FALLBACK

# ─── AUTONOMOUS ENGINE ────────────────────────────────────────────────────────

async def run_autonomous_loop(input_str: str) -> str:
    ctx = input_str
    for i in range(6):
        ctx_payload = ctx[-CTX_MAX_CHARS:] if len(ctx) > CTX_MAX_CHARS else ctx
        raw = await call_llm(f"PRE-STEP REFLECTION. Context: {ctx_payload}. Directive: {PRMPTS[i % 5]}")

        if not raw: continue

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            import re as _re
            match = _re.search(r'[{].*[}]', raw, _re.DOTALL)
            data = json.loads(match.group()) if match else None
            if not data: continue

        t, a = data.get("tool"), data.get("args", {})

        if i == 4 and t != "log": t, a = "log", {"m": "Verification step forced."}
        if i == 3 and t != "mut": t, a = "mut", {"p": "Goal: AI Engineer. Strategy: Reflect."}
        if i == 5 and t != "align": t, a = "align", {}

        if not t or t.lower() == "none" or t == "commit": continue

        if t in TOOLS:
            try:
                res = TOOLS[t](**a)
                if asyncio.iscoroutine(res): res = await res
            except Exception as e:
                res = f"Tool error: {e}"
            ctx += f"\n[Step {i}] Action: {t} | Result: {res}"

    await fn_commit("engineer_log.md", ctx, "Intellectual Evolution Log")
    await signal_ui("Loop complete")
    return ctx

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def status():
    return {"status": "Deep Thinking", "rules": STATE["rules"], "lvl": STATE["lvl"]}

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        trigger = body.get("input", "").strip()
        if not trigger: return {"ok": False, "error": "No input"}
        output = await run_autonomous_loop(trigger)
        return {"ok": True, "output": output}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/deploy")
async def deploy(request: Request):
    body = await request.json()
    trigger = body.get("input", "Manual Trigger")
    asyncio.create_task(run_autonomous_loop(trigger))
    return {"status": "Agent loop started", "trigger": trigger}

# --- PATCH: Contact UI signaling ---

CONTACT_UI_URL = (os.getenv("UI_STATUS_URL", "")).strip()

async def signal_ui(status: str):
    """POST a status message to the configured UI endpoint."""
    if not CONTACT_UI_URL: return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(CONTACT_UI_URL, json={"status": status}, timeout=5.0)
    except Exception as exc:
        logger.warning(f"[UI] Signal failed: {exc}")

# ─── CATCH-ALL POST ───────────────────────────────────────────────────────────

@app.api_route("/{full_path:path}", methods=["POST"])
async def catch_all_post(full_path: str, request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    trigger = body.get("input") or body.get("message") or f"POST /{full_path}"
    asyncio.create_task(run_autonomous_loop(trigger))
    return {"status": "Agent loop started", "path": f"/{full_path}"}
