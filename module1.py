# module1.py: Core Server and Middleware (OpenRouter Edition)

import asyncio
import os, json, time, logging
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# =========================
# Logging
# =========================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AgentServer")

# =========================
# Middleware
# =========================

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
        return await call_next(request)

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse({"error": "Internal server error", "detail": str(exc)}, status_code=500)

# =========================
# FastAPI App
# =========================

app = FastAPI(title="Autonomous Agent", version="3.0")

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

# =========================
# Environment
# =========================

OPENROUTER_API_KEY = (os.getenv("OPENROUTER_API_KEY") or "").strip()

STATE = {
    "rules": "Goal: AI Engineer. Strategy: Deep Reflection over Speed.",
    "lvl": 1,
}

CTX_MAX_CHARS = int(os.getenv("CTX_MAX_CHARS", 8000))

# =========================
# JSON Enforcement
# =========================

JSON_ENFORCEMENT = (
    "Always respond with a single valid JSON object only – "
    "no markdown, no prose, no code fences. "
    'Schema: {{"tool": "<name>", "args": {{}}, "thought": "<reasoning>"}}.'
)

SYSTEM_PROMPT_TEMPLATE = (
    "{{rules}}. "
    "You MUST respond with a single valid JSON object and nothing else – "
    "no markdown, no prose, no code fences. "
    "Your entire response must be parseable by json.loads(). "
    'Schema: {{"tool": "<name>", "args": {{}}, "thought": "<reasoning>"}}. '
    "Valid tools: env(k), log(m), math(e), fmt(d), chk(g), ui(d), mut(p), "
    "pip(package), lc(tool,input), read(path), propose_patch(instruction), "
    "apply_patch(), align(), create_module(filename,code,description), "
    "add_tool(name,code), add_step(prompt,position), list_tools(), list_steps(), "
    "reload_prompts(), run_tests(filename), create_test(filename,code,description). "
    "Use exactly these argument names. Do not add extra fields."
)

# =========================
# OpenRouter Rate Limiting
# =========================

OPENROUTER_LOCK = asyncio.Lock()
OPENROUTER_SEMAPHORE = asyncio.Semaphore(3)
OPENROUTER_CALL_TIMES: list[float] = []
OPENROUTER_RPM_LIMIT = int(os.getenv("OPENROUTER_RPM_LIMIT", 40))

FALLBACK = '{"tool": "log", "args": {"m": "API Overload"}, "thought": "retry"}'

# =========================
# OpenRouter LLM Call
# =========================

async def call_llm(prompt: str) -> str:
    async with OPENROUTER_SEMAPHORE:
        async with OPENROUTER_LOCK:
            now = time.time()
            OPENROUTER_CALL_TIMES[:] = [t for t in OPENROUTER_CALL_TIMES if now - t < 60]

            if len(OPENROUTER_CALL_TIMES) >= OPENROUTER_RPM_LIMIT:
                wait = 60 - (now - OPENROUTER_CALL_TIMES[0])
                logger.warning(f"[OpenRouter] RPM cap – waiting {wait:.1f}s")
                await asyncio.sleep(wait)

            OPENROUTER_CALL_TIMES.append(time.time())

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://localhost",
                        "X-Title": "Autonomous Agent",
                    },
                    json={
                        "model": "openrouter/auto",
                        "messages": [
                            {
                                "role": "system",
                                "content": SYSTEM_PROMPT_TEMPLATE.format(rules=STATE["rules"]) + JSON_ENFORCEMENT,
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 512,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=30.0,
                )

                resp = response.json()

            if "choices" not in resp:
                logger.error(f"[OpenRouter] Invalid response: {resp}")
                return FALLBACK

            usage = resp.get("usage", {})
            logger.info(f"[OpenRouter] tokens={usage.get('total_tokens', 0)}")

            return resp["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"[OpenRouter] call failed: {e}", exc_info=True)
            return FALLBACK

# =========================
# Routes
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def status():
    return {"status": "Deep Thinking", "rules": STATE["rules"], "lvl": STATE["lvl"]}

@app.post("/chat")
async def chat(request: Request):
    from module3 import run_autonomous_loop

    try:
        body = await request.json()
        trigger = body.get("input", "").strip()

        if not trigger:
            return {"ok": False, "error": "No input provided"}

        output = await run_autonomous_loop(trigger)
        return {"ok": True, "output": output}

    except Exception as e:
        logger.error(f"[Chat] {e}", exc_info=True)
        return {"ok": False, "error": str(e)}
