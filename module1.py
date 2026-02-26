# module1.py: Core Server and Middleware
import asyncio
import os, json, base64, time, logging, subprocess, sys
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from typing import Dict, Any, Callable
from google import genai
# The client will automatically find your 'application-default' credentials
client = genai.Client(vertexai=True, project="108183293108206939767", location="us-central1")




logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AgentServer")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        logger.info(f"-> {request.method} {request.url.path} | Client: {request.client.host}")
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"<- {response.status_code} | {elapsed:.1f}ms")
        return response

RATE_STORE: dict[str, list[float]] = {}
RATE_LIMIT  = int(os.getenv("RATE_LIMIT", 20))
RATE_WINDOW = int(os.getenv("RATE_WINDOW", 60))

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip  = request.client.host
        now = time.time()
        hits = [t for t in RATE_STORE.get(ip, []) if now - t < RATE_WINDOW]
        if len(hits) >= RATE_LIMIT:
            return JSONResponse({"error": "Rate limit exceeded"}, status_code=429)
        hits.append(now)
        RATE_STORE[ip] = hits
        stale = [k for k, v in RATE_STORE.items() if v and now - v[-1] > RATE_WINDOW * 2]
        for k in stale:
            del RATE_STORE[k]
        return await call_next(request)

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
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TrustedHostMiddleware, allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(","))

K = (os.getenv("GEMINI_API_KEY") or "").strip()
T = (os.getenv("GH_TOKEN") or "").strip()
R = (os.getenv("REPO_PATH") or "").strip()
STATE = {"rules": "Goal: AI Engineer. Strategy: Deep Reflection over Speed.", "lvl": 1}
CTX_MAX_CHARS = int(os.getenv("CTX_MAX_CHARS", 8000))

CONTACT_UI_URL = (os.getenv("UI_STATUS_URL", "")).strip()

async def signal_ui(status: str):
    if not CONTACT_UI_URL:
        return
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(CONTACT_UI_URL, json={"status": status}, timeout=5.0)
            if resp.status_code != 200:
                logger.warning(f"[UI] Signal returned HTTP {resp.status_code}")
            else:
                logger.info(f"[UI] Signaled: {status}")
    except Exception as exc:
        logger.warning(f"[UI] Signal failed (non-fatal): {exc}")

JSON_ENFORCEMENT = (
    " Always respond with a single valid JSON object only – "
    "no markdown, no prose, no code fences. "
    'Schema: {{"tool": "<n>", "args": {}, "thought": "<reasoning>"}}.'
)

GEMINI_RATE_LOCK  = asyncio.Lock()
GEMINI_SEMAPHORE  = asyncio.Semaphore(3)

GEMINI_CALL_TIMES: list[float]             = []
GEMINI_TOKEN_LOG:  list[tuple[float, int]] = []
GEMINI_DAY_CALLS:  list[float]             = []

GEMINI_RPM_LIMIT = 25
GEMINI_TPM_LIMIT = 14000
GEMINI_RPD_LIMIT = 14000

FALLBACK = '{{"tool": "log", "args": {"m": "API Overload"}, "thought": "retry"}}'

SYSTEM_PROMPT_TEMPLATE = (
    "{rules}. "
    "You MUST respond with a single valid JSON object and nothing else – "
    "no markdown, no prose, no code fences. "
    "Your entire response must be parseable by json.loads(). "
    'Schema: {{"tool": "<name>", "args": {}, "thought": "<reasoning>"}}. '
    "Valid tools: env(k), log(m), math(e), fmt(d), chk(g), ui(d), mut(p), "
    "pip(package), lc(tool,input), read(path), propose_patch(instruction), "
    "apply_patch(), align(), create_module(filename,code,description), "
    "add_tool(name,code), add_step(prompt,position), list_tools(), list_steps(), "
    "reload_prompts(), run_tests(filename), create_test(filename,code,description), "
    "autonomous_code_dev(requirements). "
    "Use exactly these argument names. Do not add extra fields."
)


client = genai.Client(api_key=K)

response = client.models.generate_content(
    model='gemma-3-27b-it', 
    contents='Hello!'
)


async def call_llm(p) -> str:
    async with GEMINI_SEMAPHORE:
        async with GEMINI_RATE_LOCK:
            now = time.time()
            GEMINI_CALL_TIMES[:] = [t for t in GEMINI_CALL_TIMES if now - t < 60]
            GEMINI_TOKEN_LOG[:]  = [(t, tk) for t, tk in GEMINI_TOKEN_LOG if now - t < 60]
            GEMINI_DAY_CALLS[:]  = [t for t in GEMINI_DAY_CALLS if now - t < 86400]

            if len(GEMINI_DAY_CALLS) >= GEMINI_RPD_LIMIT:
                wait = 86400 - (now - GEMINI_DAY_CALLS[0])
                logger.warning(f"[Gemini] RPD cap – waiting {wait:.0f}s")
                await asyncio.sleep(wait)

            if len(GEMINI_CALL_TIMES) >= GEMINI_RPM_LIMIT:
                wait = 60 - (now - GEMINI_CALL_TIMES[0])
                logger.warning(f"[Gemini] RPM cap – waiting {wait:.1f}s")
                await asyncio.sleep(wait)

            tokens_used = sum(tk for _, tk in GEMINI_TOKEN_LOG)
            if tokens_used >= GEMINI_TPM_LIMIT:
                wait = 60 - (now - GEMINI_TOKEN_LOG[0][0])
                logger.warning(f"[Gemini] TPM cap ({tokens_used}) – waiting {wait:.1f}s")
                await asyncio.sleep(wait)

            ts = time.time()
            GEMINI_CALL_TIMES.append(ts)
            GEMINI_DAY_CALLS.append(ts)

        await asyncio.sleep(1.5)

        try:
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(rules=STATE["rules"]) + JSON_ENFORCEMENT
            response = await GEMINI_MODEL.generate_content_async(
                [system_prompt, {p}],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=512,
                    response_mime_type="application/json"
                )
            )
            content = response.text
            # Approximate token count
            total_tokens = len(content.split()) * 2
            GEMINI_TOKEN_LOG.append((time.time(), total_tokens))
            logger.info(f"[Gemini] tokens≈{total_tokens} | TPM={sum(tk for _, tk in GEMINI_TOKEN_LOG)} | RPD={len(GEMINI_DAY_CALLS)}")
            return content
        except Exception as e:
            logger.error(f"[Gemini] call failed: {e}", exc_info=True)
            GEMINI_TOKEN_LOG.append((time.time(), 500))
            return FALLBACK

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def status():
    return {"status": "Deep Thinking", "rules": STATE["rules"], "lvl": STATE["lvl"]}

@app.get("/introspect")
def introspect():
    from module2 import TOOLS
    from module3 import PRMPTS
    return {
        "tools": sorted(TOOLS.keys()),
        "steps": [{"index": i, "prompt": p[:100]} for i, p in enumerate(PRMPTS)],
    }

@app.post("/chat")
async def chat(request: Request):
    from module3 import run_autonomous_loop
    try:
        body    = await request.json()
        trigger = body.get("input", "").strip()
        if not trigger:
            return {"ok": False, "error": "No input provided"}
        output = await run_autonomous_loop(trigger)
        return {"ok": True, "output": output}
    except Exception as e:
        logger.error(f"[Chat] {e}", exc_info=True)
        return {"ok": False, "error": str(e)}

@app.post("/deploy")
async def deploy(request: Request):
    from module3 import run_autonomous_loop
    body    = await request.json()
    trigger = body.get("input", "Manual Trigger via /deploy")
    asyncio.create_task(run_autonomous_loop(trigger))
    return {"status": "Agent loop started", "trigger": trigger}

@app.post("/tools/add")
async def api_add_tool(request: Request):
    from module2 import fn_add_tool
    body = await request.json()
    result = fn_add_tool(name=body.get("name",""), code=body.get("code",""))
    return {"result": result}

@app.post("/steps/add")
async def api_add_step(request: Request):
    from module2 import fn_add_step
    body = await request.json()
    result = fn_add_step(prompt=body.get("prompt",""), position=body.get("position"))
    return {"result": result}

@app.post("/modules/create")
async def api_create_module(request: Request):
    from module2 import fn_create_module
    body = await request.json()
    result = await fn_create_module(
        filename=body.get("filename",""),
        code=body.get("code",""),
        description=body.get("description",""),
    )
    return {"result": result}

@app.post("/tests/create")
async def api_create_test(request: Request):
    from module2 import fn_create_test
    body = await request.json()
    result = await fn_create_test(filename=body.get("filename",""), code=body.get("code",""), description=body.get("description",""))
    return {"result": result}

@app.post("/tests/run")
async def api_run_tests(request: Request):
    from module2 import fn_run_tests
    body = await request.json()
    result = await fn_run_tests(filename=body.get("filename",""))
    return {"result": result}

@app.post("/prompts/reload")
async def api_reload_prompts():
    from module2 import reload_prompts_from_agents_md
    updated = await reload_prompts_from_agents_md()
    from module3 import PRMPTS
    return {"updated": updated, "steps": len(PRMPTS), "prompts": PRMPTS}

@app.get("/prompts")
async def api_get_prompts():
    from module3 import PRMPTS
    return {"steps": len(PRMPTS), "prompts": PRMPTS}

@app.api_route("/{full_path:path}", methods=["POST"])
async def catch_all_post(full_path: str, request: Request):
    from module3 import run_autonomous_loop
    try:
        body = await request.json()
    except Exception:
        body = {}
    trigger = (
        body.get("input") or body.get("message") or body.get("text")
        or body.get("prompt") or (json.dumps(body) if body else f"POST /{full_path}")
    )
    logger.info(f"[Catch-All] /{full_path} → {trigger[:80]}")
    asyncio.create_task(run_autonomous_loop(trigger))
    return {"status": "Agent loop started", "path": f"/{full_path}", "trigger": trigger}
