import os
import json
import base64
import time
import logging
import subprocess
import sys
import requests
from fastapi import FastAPI, Request
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
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"→ {request.method} {request.url.path} | Client: {client_host}")
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"← {response.status_code} | {elapsed:.1f}ms")
        return response

# ─── MIDDLEWARE: Rate Limiter (per IP, in-memory) ─────────────────────────────
RATE_STORE: dict[str, list[float]] = {}
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "20"))   # requests
RATE_WINDOW = int(os.getenv("RATE_WINDOW", "60"))  # seconds

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        hits = [t for t in RATE_STORE.get(ip, []) if now - t < RATE_WINDOW]
        if len(hits) >= RATE_LIMIT:
            return JSONResponse(content={"error": "Rate limit exceeded"}, status_code=429)
        hits.append(now)
        RATE_STORE[ip] = hits
        return await call_next(request)

# ─── GITHUB COMMIT FUNCTION ───────────────────────────────────────────────────
def github_commit(file_path: str, file_content: str):
    """
    Commit (create or update) a file in the configured GitHub repository.
    Expects the following environment variables to be set:
        GITHUB_TOKEN   – personal access token with repo scope
        GITHUB_OWNER   – repository owner (user or organization)
        GITHUB_REPO    – repository name
    """
    token = os.getenv("GITHUB_TOKEN")
    owner = os.getenv("GITHUB_OWNER")
    repo = os.getenv("GITHUB_REPO")

    if not all([token, owner, repo]):
        logger.error("GitHub credentials not fully configured; skipping commit.")
        return

    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Get SHA if file already exists
    sha = None
    get_resp = requests.get(api_url, headers=headers)
    if get_resp.status_code == 200:
        sha = get_resp.json().get("sha")

    data = {
        "message": f"Update {file_path}",
        "content": base64.b64encode(file_content.encode()).decode(),
        "branch": "main"
    }
    if sha:
        data["sha"] = sha

    put_resp = requests.put(api_url, headers=headers, json=data)
    if put_resp.status_code in (200, 201):
        logger.info(f"Successfully committed {file_path} to GitHub.")
    else:
        logger.error(f"Failed to commit {file_path}: {put_resp.status_code} {put_resp.text}")

# ─── PERSISTENCE HELPERS ─────────────────────────────────────────────────────
def persist_engineer_state(state: dict):
    """Write state to engineer_state.json and push to GitHub."""
    try:
        json_str = json.dumps(state, indent=2)
        with open("engineer_state.json", "w", encoding="utf-8") as f:
            f.write(json_str)
        github_commit("engineer_state.json", json_str)
    except Exception as e:
        logger.exception(f"Error persisting engineer state: {e}")

def persist_engineer_log(entry: str):
    """Append an entry to engineer_log.md and push to GitHub."""
    try:
        with open("engineer_log.md", "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n")
        # Read full file for commit
        with open("engineer_log.md", "r", encoding="utf-8") as f:
            full_content = f.read()
        github_commit("engineer_log.md", full_content)
    except Exception as e:
        logger.exception(f"Error persisting engineer log: {e}")

# ─── FASTAPI APPLICATION ─────────────────────────────────────────────────────
app = FastAPI()

@app.on_event("startup")
async def on_startup():
    logger.info("Agent server started")
    # Signal UI that the agent is operational
    try:
        requests.get("http://localhost:8000/status")
    except Exception:
        logger.warning("Unable to reach UI endpoint during startup.")

@app.get("/status")
async def status():
    return {"status": "operational"}

# Example endpoint that demonstrates persistence usage
@app.post("/update_state")
async def update_state(payload: dict):
    persist_engineer_state(payload)
    persist_engineer_log(f"State updated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return {"result": "state persisted"}

# ─── MIDDLEWARE STACK ────────────────────────────────────────────────────────
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)