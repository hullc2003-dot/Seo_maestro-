import os, json, base64, time, asyncio, logging, subprocess, sys
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format=”%(asctime)s | %(levelname)s | %(message)s”)
logger = logging.getLogger(“AgentServer”)

# ─── MIDDLEWARE: Request Logger ───────────────────────────────────────────────

class LoggingMiddleware(BaseHTTPMiddleware):
async def dispatch(self, request: Request, call_next):
start = time.perf_counter()
logger.info(f”→ {request.method} {request.url.path} | Client: {request.client.host}”)
response = await call_next(request)
elapsed = (time.perf_counter() - start) * 1000
logger.info(f”← {response.status_code} | {elapsed:.1f}ms”)
return response

# ─── MIDDLEWARE: Rate Limiter (per IP, in-memory) ─────────────────────────────

RATE_STORE: dict[str, list[float]] = {}
RATE_LIMIT  = int(os.getenv(“RATE_LIMIT”,  20))   # requests
RATE_WINDOW = int(os.getenv(“RATE_WINDOW”, 60))   # seconds

class RateLimitMiddleware(BaseHTTPMiddleware):
async def dispatch(self, request: Request, call_next):
ip  = request.client.host
now = time.time()
hits = [t for t in RATE_STORE.get(ip, []) if now - t < RATE_WINDOW]
if len(hits) >= RATE_LIMIT:
return JSONResponse({“error”: “Rate limit exceeded”}, status_code=429)
hits.append(now)
RATE_STORE[ip] = hits
# FIX FLOW-2: evict IPs whose last hit is older than 2× the window
stale = [k for k, v in RATE_STORE.items() if v and now - v[-1] > RATE_WINDOW * 2]
for k in stale:
del RATE_STORE[k]
return await call_next(request)

# ─── MIDDLEWARE: Error Handler ────────────────────────────────────────────────

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
async def dispatch(self, request: Request, call_next):
try:
return await call_next(request)
except Exception as exc:
logger.error(f”Unhandled exception: {exc}”, exc_info=True)
return JSONResponse({“error”: “Internal server error”, “detail”: str(exc)}, status_code=500)

app = FastAPI(title=“Autonomous Agent”, version=“2.0”)

# ─── REGISTER MIDDLEWARES (order matters: first added = outermost) ────────────

app.add_middleware(ErrorHandlerMiddleware)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)

app.add_middleware(
CORSMiddleware,
allow_origins=os.getenv(“CORS_ORIGINS”, “*”).split(”,”),
allow_methods=[”*”],
allow_headers=[”*”],
)
app.add_middleware(
TrustedHostMiddleware,
allowed_hosts=os.getenv(“ALLOWED_HOSTS”, “*”).split(”,”),
)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

K = (os.getenv(“GROQ_API_KEY”) or “”).strip()
T = (os.getenv(“GH_TOKEN”)     or “”).strip()
R = (os.getenv(“REPO_PATH”)    or “”).strip()
STATE = {“rules”: “Goal: AI Engineer. Strategy: Deep Reflection over Speed.”, “lvl”: 1}

# ─── CONTEXT TRUNCATION LIMIT (chars) ─────────────────────────────────────────

CTX_MAX_CHARS = int(os.getenv(“CTX_MAX_CHARS”, 8000))

# ─── UI SIGNAL (defined early so fn_commit can reference it) ──────────────────

# FIX FLOW-1: moved above fn_commit to make the dependency explicit

CONTACT_UI_URL = (os.getenv(“UI_STATUS_URL”, “”)).strip()

async def signal_ui(status: str):
“”“POST a status message to the configured UI endpoint (UI_STATUS_URL env var).”””
if not CONTACT_UI_URL:
return
try:
async with httpx.AsyncClient() as client:
resp = await client.post(CONTACT_UI_URL, json={“status”: status}, timeout=5.0)
if resp.status_code != 200:
logger.warning(f”[UI] Signal returned HTTP {resp.status_code}”)
else:
logger.info(f”[UI] Signaled: {status}”)
except Exception as exc:
logger.warning(f”[UI] Signal failed (non-fatal): {exc}”)

# ─── TOOLS ───────────────────────────────────────────────────────────────────

def fn_1_env(k=””, **kwargs): return os.getenv(k, “Null”)

def fn_2_log(m=None, **kwargs):
msg = m or json.dumps(kwargs) or “Log recorded”
logger.info(f”[Reflect]: {msg}”)
return “Log recorded”

# FIX FLOW-3: accept both ‘e’ and common aliases so the LLM isn’t brittle

def fn_3_math(e=None, expression=None, expr=None, **kwargs):
formula = e or expression or expr or “”
if not formula:
return “Math Err: no expression”
try:
from simpleeval import simple_eval
return simple_eval(formula)
except Exception:
return “Math Err”

def fn_4_fmt(d=””, **kwargs): return f”### ANALYSIS ###\n{d or json.dumps(kwargs)}”
def fn_5_chk(g=””, **kwargs): return f”Goal Alignment: {g or json.dumps(kwargs)}”
def fn_6_ui(d=””,  **kwargs): return f”UI_UPDATE: {d or json.dumps(kwargs)}”

def fn_7_mut(p=””, **kwargs):
new_rules = p or kwargs.get(“rules”) or kwargs.get(“ruleset”) or “”
if not new_rules or not new_rules.strip():
logger.warning(”[mut] Rejected empty ruleset – STATE[‘rules’] unchanged”)
return “Mut rejected: empty ruleset”
STATE[“rules”] = new_rules.strip()
return “Core Rules Redefined”

JSON_ENFORCEMENT = (
“ Always respond with a single valid JSON object only – “
“no markdown, no prose, no code fences. “
‘Schema: {“tool”: “<n>”, “args”: {}, “thought”: “<reasoning>”}.’
)

# ─── GITHUB COMMIT ────────────────────────────────────────────────────────────

async def fn_commit(path, content, msg):
try:
if not T:
logger.error(”[Commit] GH_TOKEN env var is not set”)
return “Save_Failed: no GH_TOKEN”
if not R:
logger.error(”[Commit] REPO_PATH env var is not set”)
return “Save_Failed: no REPO_PATH”
headers = {“Authorization”: f”token {T}”, “User-Agent”: “AIEngAgent”}
async with httpx.AsyncClient() as client:
get_resp = await client.get(
f”https://api.github.com/repos/{R}/contents/{path}”,
headers=headers
)
get_data = get_resp.json()
if get_resp.status_code not in (200, 404):
logger.error(f”[Commit] GET failed {get_resp.status_code}: {get_data}”)
return f”Save_Failed: GET {get_resp.status_code}”
sha = get_data.get(“sha”, “”)
put_resp = await client.put(
f”https://api.github.com/repos/{R}/contents/{path}”,
headers=headers,
json={
“message”: msg,
“content”: base64.b64encode(content.encode()).decode(),
“sha”: sha,
},
)
put_data = put_resp.json()
if put_resp.status_code not in (200, 201):
logger.error(f”[Commit] PUT failed {put_resp.status_code}: {put_data}”)
return f”Save_Failed: PUT {put_resp.status_code}”
saved = f”Saved_{put_resp.status_code}”
# FIX BUG-7: asyncio.create_task() instead of deprecated ensure_future()
asyncio.create_task(signal_ui(f”Committed {path}”))
return saved
except Exception as e:
logger.error(f”[Commit] Exception: {e}”, exc_info=True)
return “Save_Failed”

# ─── LANGCHAIN BOOTSTRAP ──────────────────────────────────────────────────────

def _ensure_pkg(pkg, import_as=None):
“”“Install a package at runtime if missing.”””
name = import_as or pkg.split(”[”)[0]
try:
# FIX BUG-1: **import** was mangled to **import** by markdown renderer
**import**(name)
except ImportError:
logger.info(f”[Bootstrap] Installing {pkg}…”)
# FIX BUG-2: em-dash “–quiet” → double-hyphen “–quiet”
subprocess.check_call([sys.executable, “-m”, “pip”, “install”, pkg, “–quiet”])

def fn_8_pip(package=””, **kwargs):
“”“Install a Python package so it can be used in future tool calls.”””
pkg = package or kwargs.get(“pkg”, “”)
if not pkg:
return “pip_err: no package name”
try:
_ensure_pkg(pkg)
return f”Installed: {pkg}”
except Exception as e:
logger.error(f”[pip] {e}”)
return f”pip_err: {e}”

def fn_9_lc_tool(tool=””, input=””, **kwargs):
“”“Run a LangChain community tool by name (e.g. ‘ddg-search’, ‘wikipedia’).”””
tool = tool or kwargs.get(“name”, “”)
inp  = input or kwargs.get(“query”, “”) or kwargs.get(“input”, “”)
if not tool or not inp:
return “lc_err: need tool and input”
try:
_ensure_pkg(“langchain-community”, “langchain_community”)
_ensure_pkg(“duckduckgo-search”,   “duckduckgo_search”)
if tool in (“ddg-search”, “search”, “web”):
from langchain_community.tools import DuckDuckGoSearchRun
return DuckDuckGoSearchRun().run(inp)[:1500]
if tool in (“wikipedia”, “wiki”):
_ensure_pkg(“wikipedia”, “wikipedia”)
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run(inp)[:1500]
return f”lc_err: unknown tool ‘{tool}’”
except Exception as e:
logger.error(f”[lc_tool] {e}”, exc_info=True)
return f”lc_err: {e}”

# ─── GITHUB FILE READER ───────────────────────────────────────────────────────

async def fn_read_github(path=””, **kwargs):
“”“Read any file from the repo. Use path=‘agents.md’ or path=‘main.py’.”””
path = path or kwargs.get(“file”, “”)
if not path:
return “read_err: no path”
if not T or not R:
return “read_err: missing GH_TOKEN or REPO_PATH”
try:
headers = {“Authorization”: f”token {T}”, “User-Agent”: “AIEngAgent”}
async with httpx.AsyncClient() as client:
resp = await client.get(
f”https://api.github.com/repos/{R}/contents/{path}”,
headers=headers, timeout=15.0,
)
data = resp.json()
if resp.status_code != 200:
return f”read_err: {resp.status_code} {data.get(‘message’,’’)}”
content = base64.b64decode(data[“content”]).decode(“utf-8”, errors=“replace”)
logger.info(f”[ReadGitHub] {path} ({len(content)} chars)”)
return content[:6000]
except Exception as e:
logger.error(f”[ReadGitHub] {e}”, exc_info=True)
return f”read_err: {e}”

# ─── SELF-PATCH TOOLS ─────────────────────────────────────────────────────────

# FIX BUG-6 / BUG-5: use a consistent, minimal patch header so that

# fn_propose_patch and fn_apply_patch agree on the marker format.

PATCH_HEADER_PREFIX = “# — PATCH:”   # hyphens only, no em-dash

async def fn_propose_patch(instruction=””, **kwargs):
“”“Ask the LLM to generate a targeted code patch aligning main.py with agents.md.
Writes result to main_patch.py on GitHub for review before applying.”””
instruction = instruction or kwargs.get(“desc”, “”)
if not instruction:
return “patch_err: no instruction”
current_code = await fn_read_github(“main.py”)
agents_spec  = await fn_read_github(“agents.md”)
if “read_err” in current_code:
return f”patch_err: could not read main.py – {current_code}”
if “read_err” in agents_spec:
return f”patch_err: could not read agents.md – {agents_spec}”

```
prompt = (
    f"You are a Python engineer making a MINIMAL addition to an existing file.\n\n"
    f"agents.md spec (excerpt):\n{agents_spec[:600]}\n\n"
    f"Current main.py tail (last 800 chars):\n{current_code[-800:]}\n\n"
    f"Instruction: {instruction}\n\n"
    "Write ONLY the new Python code to ADD (functions, constants, or route handlers). "
    "Do NOT rewrite the whole file. Do NOT include imports that already exist. "
    f"Output MUST start with exactly: {PATCH_HEADER_PREFIX} <description> ---\n"
    "No markdown fences. Raw Python only. Max 60 lines."
)
try:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {K}", "Content-Type": "application/json"},
            json={
                # FIX BUG-4: valid Groq model name
                "model": "compound-beta",
                "messages": [
                    {"role": "system", "content": "You are a Python code generator. Output raw Python only."},
                    {"role": "user",   "content": prompt},
                ],
                "max_tokens": 1024,
            },
            timeout=60.0,
        )
    rdata2 = resp.json()
    if "choices" not in rdata2:
        err = rdata2.get("error", rdata2)
        logger.error(f"[ProposePatch] Groq call failed: {err}")
        return f"patch_err: groq={err}"
    proposed = rdata2["choices"][0]["message"]["content"].strip()
    if proposed.startswith("```"):
        proposed = proposed.split("\n", 1)[1].rsplit("```", 1)[0]
    # FIX BUG-8: commit path and return message now agree
    result = await fn_commit("main_patch.py", proposed, f"[AutoPatch] {instruction[:80]}")
    logger.info(f"[ProposePatch] Patch saved to main_patch.py — review before applying")
    return f"Proposed patch committed as main_patch.py — {result}"
except Exception as e:
    logger.error(f"[ProposePatch] {e}", exc_info=True)
    return f"patch_err: {e}"
```

async def fn_apply_patch(**kwargs):
“”“Append main_patch.py content into main.py on GitHub.”””
patch = await fn_read_github(“main_patch.py”)
if “read_err” in patch:
return f”apply_err: {patch}”
# FIX BUG-6: check for hyphens-only prefix, matching what fn_propose_patch writes
if PATCH_HEADER_PREFIX not in patch:
return f”apply_err: patch missing header ‘{PATCH_HEADER_PREFIX}’ — unsafe to apply”
current = await fn_read_github(“main.py”)
if “read_err” in current:
return f”apply_err: could not read main.py – {current}”

```
# FIX BUG-5: use a simple, stable marker that actually exists in main.py
# rather than embedding the entire signal_ui source as the marker string.
INSERT_MARKER = "# ─── CATCH-ALL POST"
if INSERT_MARKER in current:
    updated = current.replace(INSERT_MARKER, patch.strip() + "\n\n\n" + INSERT_MARKER, 1)
else:
    updated = current + "\n\n" + patch.strip()

# FIX FLOW-6: widen sanity check to things less likely to be renamed
if "FastAPI" not in updated or "run_autonomous_loop" not in updated:
    return "apply_err: merged file failed sanity check — aborting"

result = await fn_commit("main.py", updated, "[AutoApply] Append patch to main.py")
logger.info(f"[ApplyPatch] {result}")
return f"main.py patched — {result}"
```

# ─── AGENTS.MD ALIGNMENT STEP ────────────────────────────────────────────────

async def fn_align_with_spec(**kwargs):
“”“Read agents.md, compare with current capabilities, log gaps, propose patch.”””
agents_spec  = await fn_read_github(“agents.md”)
if “read_err” in agents_spec:
logger.warning(f”[Align] Could not read agents.md: {agents_spec}”)
return f”align_skipped: {agents_spec}”
current_code = await fn_read_github(“main.py”)
gap_prompt = (
f”agents.md specification:\n{agents_spec[:800]}\n\n”
f”Current main.py (truncated):\n{current_code[:600]}\n\n”
“In one sentence, what is the single most important capability in agents.md “
“that is missing or incomplete in main.py right now?”
)
try:
await asyncio.sleep(8)
async with httpx.AsyncClient() as client:
resp = await client.post(
“https://api.groq.com/openai/v1/chat/completions”,
headers={“Authorization”: f”Bearer {K}”, “Content-Type”: “application/json”},
json={
# FIX BUG-4: valid Groq model name
“model”: “compound-beta”,
“messages”: [{“role”: “user”, “content”: gap_prompt}],
“max_tokens”: 200,
},
timeout=30.0,
)
rdata = resp.json()
if “choices” not in rdata:
err  = rdata.get(“error”, rdata)
logger.error(f”[Align] Groq gap-call failed: {err}”)
gap  = “agents.md exists but gap LLM call failed — retrying next loop”
fn_2_log(m=f”[Align] {gap}”)
return f”align_partial: {err}”
gap = rdata[“choices”][0][“message”][“content”].strip()
logger.info(f”[Align] Gap identified: {gap}”)
await asyncio.sleep(10)
patch_result = await fn_propose_patch(instruction=gap)
return f”Gap: {gap} | {patch_result}”
except Exception as e:
logger.error(f”[Align] {e}”, exc_info=True)
return f”align_err: {e}”

# ─── TOOL REGISTRY ────────────────────────────────────────────────────────────

TOOLS = {
“env”:           fn_1_env,
“log”:           fn_2_log,
“math”:          fn_3_math,
“fmt”:           fn_4_fmt,
“chk”:           fn_5_chk,
“ui”:            fn_6_ui,
“mut”:           fn_7_mut,
“commit”:        fn_commit,
“pip”:           fn_8_pip,
“lc”:            fn_9_lc_tool,
“read”:          fn_read_github,
“propose_patch”: fn_propose_patch,
“apply_patch”:   fn_apply_patch,
“align”:         fn_align_with_spec,
}

PRMPTS = [
“Critically analyze the current state. What is missing to reach AI Engineer status? You MUST call tool=‘chk’ with args={‘g’: ‘<your one-sentence gap summary>’}.”,
“Generate a hypothesis for a better autonomous pattern. You MUST call tool=‘log’ with args={‘m’: ‘<your hypothesis in one sentence>’}.”,
“Identify the single most critical failure point in the previous step. You MUST call tool=‘fmt’ with args={‘d’: ‘<failure point in one sentence>’}.”,
“You MUST call tool=‘mut’ with args={‘p’: ‘<one sentence ruleset>’}. Keep p under 100 characters. No markdown.”,
“Log a one-sentence final verification summary. You MUST call tool=‘log’ with args={‘m’: ‘<verification summary>’}.”,
“MANDATORY FINAL STEP – no other tool is valid here. You MUST call tool=‘align’ with args={}. Do NOT call chk, log, or any other tool.”,
]

# ─── GROQ RATE LIMITING ───────────────────────────────────────────────────────

# FIX FLOW-7: use a lock around rate-check + register to prevent race conditions

GROQ_RATE_LOCK   = asyncio.Lock()
GROQ_SEMAPHORE   = asyncio.Semaphore(3)

GROQ_CALL_TIMES: list[float]        = []
GROQ_TOKEN_LOG:  list[tuple[float, int]] = []
GROQ_DAY_CALLS:  list[float]        = []

GROQ_RPM_LIMIT = int(os.getenv(“GROQ_RPM_LIMIT”,  25))
GROQ_TPM_LIMIT = int(os.getenv(“GROQ_TPM_LIMIT”, 28_000))
GROQ_RPD_LIMIT = int(os.getenv(“GROQ_RPD_LIMIT”,  250))

# ─── LLM CALL ─────────────────────────────────────────────────────────────────

FALLBACK = ‘{“tool”: “log”, “args”: {“m”: “API Overload”}, “thought”: “retry”}’

SYSTEM_PROMPT_TEMPLATE = (
“{rules}. “
“You MUST respond with a single valid JSON object and nothing else – “
“no markdown, no prose, no code fences. “
“Your entire response must be parseable by json.loads(). “
’Schema: {{“tool”: “<name>”, “args”: {{}}, “thought”: “<reasoning>”}}. ’
“Valid tools: env(k), log(m), math(e), fmt(d), chk(g), ui(d), mut(p), “
“pip(package), lc(tool,input), read(path), propose_patch(instruction), “
“apply_patch(), align(). “
“Use exactly these argument names. Do not add extra fields.”
)

async def call_llm(p) -> str:
async with GROQ_SEMAPHORE:
# FIX FLOW-7: serialize rate-check + register under a lock
async with GROQ_RATE_LOCK:
now = time.time()
GROQ_CALL_TIMES[:] = [t for t in GROQ_CALL_TIMES if now - t < 60]
GROQ_TOKEN_LOG[:]  = [(t, tk) for t, tk in GROQ_TOKEN_LOG if now - t < 60]
GROQ_DAY_CALLS[:]  = [t for t in GROQ_DAY_CALLS if now - t < 86_400]

```
        if len(GROQ_DAY_CALLS) >= GROQ_RPD_LIMIT:
            wait = 86_400 - (now - GROQ_DAY_CALLS[0])
            logger.warning(f"[Groq throttle] RPD cap – waiting {wait:.0f}s (~{wait/3600:.1f}h)")
            await asyncio.sleep(wait)

        if len(GROQ_CALL_TIMES) >= GROQ_RPM_LIMIT:
            wait = 60 - (now - GROQ_CALL_TIMES[0])
            logger.warning(f"[Groq throttle] RPM cap – waiting {wait:.1f}s")
            await asyncio.sleep(wait)

        tokens_used = sum(tk for _, tk in GROQ_TOKEN_LOG)
        if tokens_used >= GROQ_TPM_LIMIT:
            wait = 60 - (now - GROQ_TOKEN_LOG[0][0])
            logger.warning(f"[Groq throttle] TPM cap ({tokens_used} used) – waiting {wait:.1f}s")
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
                    # FIX BUG-4: valid Groq model name
                    "model": "compound-beta",
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT_TEMPLATE.format(rules=STATE["rules"])
                                       + JSON_ENFORCEMENT,
                        },
                        {"role": "user", "content": p},
                    ],
                    # FIX FLOW-8: cap token output to protect TPM budget
                    "max_tokens": 512,
                    "response_format": {"type": "json_object"},
                },
                timeout=30.0,
            )
            resp = response.json()

        if "choices" not in resp:
            err = resp.get("error", {})
            if err.get("type") == "permissions_error":
                logger.error(f"[Groq] Permissions error: {err.get('message')}")
                raise RuntimeError(f"Groq permissions error: {err.get('message')}")
            logger.error(f"[Groq] No 'choices' in response: {resp}")
            GROQ_TOKEN_LOG.append((time.time(), 0))
            return FALLBACK

        usage        = resp.get("usage", {})
        total_tokens = usage.get("total_tokens", 500)
        GROQ_TOKEN_LOG.append((time.time(), total_tokens))
        logger.info(
            f"[Groq] tokens={total_tokens} | "
            f"TPM_window={sum(tk for _, tk in GROQ_TOKEN_LOG)} | "
            f"RPD={len(GROQ_DAY_CALLS)}/250"
        )
        return resp["choices"][0]["message"]["content"]

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"[Groq] API call failed: {e}", exc_info=True)
        GROQ_TOKEN_LOG.append((time.time(), 500))
        return FALLBACK
```

# ─── AUTONOMOUS ENGINE ────────────────────────────────────────────────────────

async def run_autonomous_loop(input_str: str) -> str:
ctx = input_str
for i in range(6):
ctx_payload = ctx[-CTX_MAX_CHARS:] if len(ctx) > CTX_MAX_CHARS else ctx

```
    # FIX BUG-3: PRMPTS[i] not PRMPTS[i % 5] — ensures step 5 gets the align prompt
    raw = await call_llm(
        f"PRE-STEP REFLECTION. Current Context: {ctx_payload}. Directive: {PRMPTS[i]}"
    )

    if not raw:
        logger.warning(f"[Loop] Step {i}: call_llm returned empty, skipping")
        continue

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        import re as _re
        match = _re.search(r'[{].*[}]', raw, _re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                logger.warning(f"[Loop] Step {i}: extracted JSON from markdown response")
            except Exception:
                data = None
        else:
            data = None
        if not data:
            if i == 3:
                safe = raw[:500].replace('"', "'").replace('\n', ' ').strip()
                data = {"tool": "mut", "args": {"p": safe}, "thought": "extracted from markdown"}
            else:
                logger.error(f"[Loop] Step {i}: failed to parse LLM response: {e} | raw={raw!r}")
                continue

    t, a = data.get("tool"), data.get("args", {})

    # Step guards
    if i == 4 and t != "log":
        logger.warning(f"[Loop] Step 4: LLM tried '{t}' — forcing log()")
        t = "log"
        a = {"m": f"Verification: steps 0-3 complete. Context: {ctx[-150:].replace(chr(10),' ')}"}

    if i == 3 and t != "mut":
        logger.warning(f"[Loop] Step 3: LLM tried '{t}' — forcing mut()")
        t = "mut"
        a = {"p": f"Goal: AI Engineer. Full analysis required. Context: {ctx[-200:].replace(chr(10),' ')}"}

    if i == 5 and t != "align":
        logger.warning(f"[Loop] Step 5: LLM tried '{t}' — forcing align()")
        t = "align"
        a = {}
    if i == 5 and t == "align":
        logger.info("[Loop] Step 5: running align() against agents.md")

    if not t or t.lower() == "none":
        logger.info(f"[Loop] Step {i}: LLM chose no tool — skipping")
        ctx += f"\n[Step {i}] No action taken | Reasoning: {data.get('thought')}"
        continue

    if t == "commit":
        logger.warning(f"[Loop] Step {i}: LLM tried 'commit' mid-loop — blocked")
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
        logger.warning(f"[Loop] Step {i}: unknown tool '{t}'")

commit_result = await fn_commit("engineer_log.md", ctx, "Intellectual Evolution Log")
logger.info(f"[Commit] {commit_result}")
await signal_ui("Loop complete")
return ctx
```

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get(”/health”)
def health():
return {“status”: “ok”}

@app.get(”/status”)
def status():
return {“status”: “Deep Thinking”, “rules”: STATE[“rules”], “lvl”: STATE[“lvl”]}

@app.post(”/chat”)
async def chat(request: Request):
try:
body    = await request.json()
trigger = body.get(“input”, “”).strip()
if not trigger:
return {“ok”: False, “error”: “No input provided”}
output = await run_autonomous_loop(trigger)
return {“ok”: True, “output”: output}
except Exception as e:
logger.error(f”[Chat] Error: {e}”, exc_info=True)
return {“ok”: False, “error”: str(e)}

@app.post(”/deploy”)
async def deploy(request: Request):
body    = await request.json()
trigger = body.get(“input”, “Manual Trigger via /deploy”)
asyncio.create_task(run_autonomous_loop(trigger))
return {“status”: “Agent loop started”, “trigger”: trigger}

# ─── CATCH-ALL POST ───────────────────────────────────────────────────────────

@app.api_route(”/{full_path:path}”, methods=[“POST”])
async def catch_all_post(full_path: str, request: Request):
try:
body = await request.json()
except Exception:
body = {}

```
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
```
