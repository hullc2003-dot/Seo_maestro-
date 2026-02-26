import os
import time
import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format=”%(asctime)s | %(levelname)s | %(message)s”)
logger = logging.getLogger(“AgentServer”)

# ─── LOGGING MIDDLEWARE ───────────────────────────────────────────────────────

class LoggingMiddleware(BaseHTTPMiddleware):
async def dispatch(self, request: Request, call_next):
start = time.perf_counter()
logger.info(f”-> {request.method} {request.url.path} | Client: {request.client.host}”)
response = await call_next(request)
elapsed = (time.perf_counter() - start) * 1000
logger.info(f”<- {response.status_code} | {elapsed:.1f}ms”)
return response

# ─── RATE LIMIT MIDDLEWARE ────────────────────────────────────────────────────

RATE_STORE: dict[str, list[float]] = {}
RATE_LIMIT  = int(os.getenv(“RATE_LIMIT”, 20))
RATE_WINDOW = int(os.getenv(“RATE_WINDOW”, 60))

class RateLimitMiddleware(BaseHTTPMiddleware):
async def dispatch(self, request: Request, call_next):
ip  = request.client.host
now = time.time()
hits = [t for t in RATE_STORE.get(ip, []) if now - t < RATE_WINDOW]
if len(hits) >= RATE_LIMIT:
return JSONResponse({“error”: “Rate limit exceeded”}, status_code=429)
hits.append(now)
RATE_STORE[ip] = hits
stale = [k for k, v in RATE_STORE.items() if v and now - v[-1] > RATE_WINDOW * 2]
for k in stale:
del RATE_STORE[k]
return await call_next(request)

# ─── ERROR HANDLER MIDDLEWARE ─────────────────────────────────────────────────

class ErrorHandlerMiddleware(BaseHTTPMiddleware):
async def dispatch(self, request: Request, call_next):
try:
return await call_next(request)
except Exception as exc:
logger.error(f”Unhandled exception: {exc}”, exc_info=True)
return JSONResponse(
{“error”: “Internal server error”, “detail”: str(exc)},
status_code=500,
)
