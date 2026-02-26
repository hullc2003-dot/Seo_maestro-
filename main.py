“””
main.py – FastAPI application, Groq LLM client, autonomous loop, and API routes.
“””

import asyncio
import json
import logging
import os
import re
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from middleware import ErrorHandlerMiddleware, LoggingMiddleware, RateLimitMiddleware
from tools import (
JSON_ENFORCEMENT,
PRMPTS,
STATE,
SYSTEM_PROMPT_TEMPLATE,
TOOLS,
fn_commit,
fn_add_step,
fn_add_tool,
fn_create_module,
fn_create_test,
fn_run_tests,
reload_prompts_from_agents_md,
signal_ui,
)

logger = logging.getLogger(“AgentServer”)

K              = (os.getenv(“GROQ_API_KEY”) or “”).strip()
CTX_MAX_CHARS  = int(os.getenv(“CTX_MAX_CHARS”, 8000))

# ─── FASTAPI APP ──────────────────────────────────────────────────────────────

app = FastAPI(title=“Autonomous Agent”, version=“2.0”)
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

# ─── GROQ RATE LIMITING ───────────────────────────────────────────────────────

GROQ_RATE_LOCK = asyncio.Lock()
GROQ_SEMAPHORE = asyncio.Semaphore(3)

GROQ_CALL_TIMES: list[float]             = []
GROQ_TOKEN_LOG:  list[tuple[float, int]] = []
GROQ_DAY_CALLS:  list[float]             = []

GROQ_RPM_LIMIT = int(os.getenv(“GROQ_RPM_LIMIT”, 25))
GROQ_TPM_LIMIT = int(os.getenv(“GROQ_TPM_LIMIT”, 28_000))
GROQ_RPD_LIMIT = int(os.getenv(“GROQ_RPD_LIMIT”, 250))

FALLBACK = ‘{“tool”: “log”, “args”: {“m”: “API Overload”}, “thought”: “retry”}’

# ─── LLM CALL ─────────────────────────────────────────────────────────────────

async def call_llm(p) -> str:
async with GROQ_SEMAPHORE:
async with GROQ_RATE_LOCK:
now = time.time()
GROQ_CALL_TIMES[:] = [t for t in GROQ_CALL_TIMES if now - t < 60]
GROQ_TOKEN_LOG[:]  = [(t, tk) for t, tk in GROQ_TOKEN_LOG if now - t < 60]
GROQ_DAY_CALLS[:]  = [t for t in GROQ_DAY_CALLS if now - t < 86_400]

```
        if len(GROQ_DAY_CALLS) >= GROQ_RPD_LIMIT:
            wait = 86_400 - (now - GROQ_DAY_CALLS[0])
            logger.warning(f"[Groq] RPD cap – waiting {wait:.0f}s")
            await asyncio.sleep(wait)

        if len(GROQ_CALL_TIMES) >= GROQ_RPM_LIMIT:
            wait = 60 - (now - GROQ_CALL_TIMES[0])
            logger.warning(f"[Groq] RPM cap – waiting {wait:.1f}s")
            await asyncio.sleep(wait)

        tokens_used = sum(tk for _, tk in GROQ_TOKEN_LOG)
        if tokens_used >= GROQ_TPM_LIMIT:
            wait = 60 - (now - GROQ_TOKEN_LOG[0][0])
            logger.warning(f"[Groq] TPM cap ({tokens_used}) – waiting {wait:.1f}s")
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
                    "model": "compound-beta",
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT_TEMPLATE.format(rules=STATE["rules"]) + JSON_ENFORCEMENT,
                        },
                        {"role": "user", "content": p},
                    ],
                    "max_tokens": 512,
                    "response_format": {"type": "json_object"},
                },
                timeout=30.0,
            )
            resp = response.json()

        if "choices" not in resp:
            err = resp.get("error", {})
            if err.get("type") == "permissions_error":
                raise RuntimeError(f"Groq permissions error: {err.get('message')}")
            logger.error(f"[Groq] No 'choices': {resp}")
            GROQ_TOKEN_LOG.append((time.time(), 0))
            return FALLBACK

        usage        = resp.get("usage", {})
        total_tokens = usage.get("total_tokens", 500)
        GROQ_TOKEN_LOG.append((time.time(), total_tokens))
        logger.info(
            f"[Groq] tokens={total_tokens} | "
            f"TPM={sum(tk for _, tk in GROQ_TOKEN_LOG)} | "
            f"RPD={len(GROQ_DAY_CALLS)}"
        )
        return resp["choices"][0]["message"]["content"]

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"[Groq] call failed: {e}", exc_info=True)
        GROQ_TOKEN_LOG.append((time.time(), 500))
        return FALLBACK
```

# ─── AUTONOMOUS ENGINE ────────────────────────────────────────────────────────

async def run_autonomous_loop(input_str: str) -> str:
ctx = input_str

```
await reload_prompts_from_agents_md()
logger.info(f"[Loop] Starting with {len(PRMPTS)} steps")

i = 0
while i < len(PRMPTS):
    ctx_payload = ctx[-CTX_MAX_CHARS:] if len(ctx) > CTX_MAX_CHARS else ctx
    ctx_snip    = ctx_payload[-1200:]
    directive   = PRMPTS[i][:400]
    user_msg    = f"STEP {i}. Context: {ctx_snip}\nDirective: {directive}"
    if len(user_msg) > 2800:
        user_msg = user_msg[-2800:]

    raw = await call_llm(user_msg)

    if not raw:
        logger.warning(f"[Loop] Step {i}: empty LLM response, skipping")
        i += 1
        continue

    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        match = re.search(r'[{].*[}]', raw, re.DOTALL)
        data  = None
        if match:
            try:
                data = json.loads(match.group())
                logger.warning(f"[Loop] Step {i}: extracted JSON from markdown")
            except Exception:
                pass
        if not data:
            align_idx = len(PRMPTS) - 1
            mut_idx   = 3
            if i == mut_idx:
                safe = raw[:500].replace('"', "'").replace('\n', ' ').strip()
                data = {"tool": "mut", "args": {"p": safe}, "thought": "extracted from markdown"}
            else:
                logger.error(f"[Loop] Step {i}: parse failed: {e} | raw={raw!r}")
                i += 1
                continue

    t, a = data.get("tool"), data.get("args", {})
    align_idx = len(PRMPTS) - 1
    mut_idx   = 3

    # Step guards
    if i == 4 and t != "log":
        logger.warning(f"[Loop] Step 4: forcing log()")
        t = "log"
        a = {"m": f"Verification: steps 0-3 complete. {ctx[-120:].replace(chr(10),' ')}"}

    if i == mut_idx and t != "mut":
        logger.warning(f"[Loop] Step {i}: forcing mut()")
        t = "mut"
        a = {"p": f"Goal: AI Engineer. Full analysis required. {ctx[-200:].replace(chr(10),' ')}"}

    if i == align_idx and t != "align":
        logger.warning(f"[Loop] Step {i}: forcing align()")
        t = "align"
        a = {}

    if not t or t.lower() == "none":
        logger.info(f"[Loop] Step {i}: no tool chosen")
        ctx += f"\n[Step {i}] No action | Reasoning: {data.get('thought')}"
        i += 1
        continue

    if t == "commit":
        logger.warning(f"[Loop] Step {i}: mid-loop commit blocked")
        ctx += f"\n[Step {i}] Commit blocked | Reasoning: {data.get('thought')}"
        i += 1
        continue

    if t in TOOLS:
        try:
            res = TOOLS[t](**a)
            if asyncio.iscoroutine(res):
                res = await res
        except Exception as exc:
            res = f"Tool error: {exc}"
            logger.error(f"[Loop] Step {i}: tool '{t}' raised: {exc}", exc_info=True)
        ctx += f"\n[Step {i}] Action: {t} | Result: {res} | Reasoning: {data.get('thought')}"
    else:
        logger.warning(f"[Loop] Step {i}: unknown tool '{t}'")
        ctx += f"\n[Step {i}] Unknown tool '{t}' | Reasoning: {data.get('thought')}"

    i += 1

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

@app.get(”/introspect”)
def introspect():
“”“Live view of registered tools and current step sequence.”””
return {
“tools”: sorted(TOOLS.keys()),
“steps”: [{“index”: i, “prompt”: p[:100]} for i, p in enumerate(PRMPTS)],
}

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
logger.error(f”[Chat] {e}”, exc_info=True)
return {“ok”: False, “error”: str(e)}

@app.post(”/deploy”)
async def deploy(request: Request):
body    = await request.json()
trigger = body.get(“input”, “Manual Trigger via /deploy”)
asyncio.create_task(run_autonomous_loop(trigger))
return {“status”: “Agent loop started”, “trigger”: trigger}

@app.post(”/tools/add”)
async def api_add_tool(request: Request):
“”“REST shortcut: POST {“name”:”…”, “code”:”…”} to register a tool.”””
body   = await request.json()
result = fn_add_tool(name=body.get(“name”, “”), code=body.get(“code”, “”))
return {“result”: result}

@app.post(”/steps/add”)
async def api_add_step(request: Request):
“”“REST shortcut: POST {“prompt”:”…”, “position”: <int|null>} to add a step.”””
body   = await request.json()
result = fn_add_step(prompt=body.get(“prompt”, “”), position=body.get(“position”))
return {“result”: result}

@app.post(”/modules/create”)
async def api_create_module(request: Request):
“”“REST shortcut: POST {“filename”:”…”, “code”:”…”, “description”:”…”} to create a module.”””
body   = await request.json()
result = await fn_create_module(
filename=body.get(“filename”, “”),
code=body.get(“code”, “”),
description=body.get(“description”, “”),
)
return {“result”: result}

@app.post(”/tests/create”)
async def api_create_test(request: Request):
body   = await request.json()
result = await fn_create_test(
filename=body.get(“filename”, “”),
code=body.get(“code”, “”),
description=body.get(“description”, “”),
)
return {“result”: result}

@app.post(”/tests/run”)
async def api_run_tests(request: Request):
body   = await request.json()
result = await fn_run_tests(filename=body.get(“filename”, “”))
return {“result”: result}

@app.post(”/prompts/reload”)
async def api_reload_prompts():
updated = await reload_prompts_from_agents_md()
return {“updated”: updated, “steps”: len(PRMPTS), “prompts”: PRMPTS}

@app.get(”/prompts”)
async def api_get_prompts():
return {“steps”: len(PRMPTS), “prompts”: PRMPTS}

# ─── CATCH-ALL POST ───────────────────────────────────────────────────────────

@app.api_route(”/{full_path:path}”, methods=[“POST”])
async def catch_all_post(full_path: str, request: Request):
try:
body = await request.json()
except Exception:
body = {}
trigger = (
body.get(“input”) or body.get(“message”) or body.get(“text”)
or body.get(“prompt”) or (json.dumps(body) if body else f”POST /{full_path}”)
)
logger.info(f”[Catch-All] /{full_path} → {trigger[:80]}”)
asyncio.create_task(run_autonomous_loop(trigger))
return {“status”: “Agent loop started”, “path”: f”/{full_path}”, “trigger”: trigger}
