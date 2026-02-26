# module2.py: Optimized Tools and Functions (Deterministic + OpenRouter Escalation)

import asyncio
import base64
import json
import logging
import subprocess
import sys
import time
import httpx
import os
import ast
import re
import hashlib
from typing import Dict, Any, Callable

from module1 import T, R, STATE, CTX_MAX_CHARS, logger, call_llm

# =========================
# LLM Guardrails + Cache
# =========================

LLM_CACHE: Dict[str, str] = {}
LLM_CALL_LIMIT = 5
LLM_CALL_COUNT = 0


async def safe_call_llm(prompt: str) -> str:
    global LLM_CALL_COUNT

    key = hashlib.sha256(prompt.encode()).hexdigest()
    if key in LLM_CACHE:
        return LLM_CACHE[key]

    if LLM_CALL_COUNT >= LLM_CALL_LIMIT:
        return "llm_guard: call limit reached"

    LLM_CALL_COUNT += 1
    response = await call_llm(prompt)

    if response:
        LLM_CACHE[key] = response

    return response or ""


# =========================
# Basic Tools
# =========================

def fn_1_env(k="", **kwargs):
    return os.getenv(k, "Null")


def fn_2_log(m=None, **kwargs):
    msg = m or json.dumps(kwargs) or "Log recorded"
    logger.info(f"[Reflect]: {msg}")
    return "Log recorded"


def fn_3_math(e=None, expression=None, expr=None, **kwargs):
    formula = e or expression or expr or ""
    if not formula:
        return "Math Err: no expression"
    try:
        from simpleeval import simple_eval
        return simple_eval(formula)
    except Exception:
        return "Math Err"


def fn_4_fmt(d="", **kwargs):
    return f"### ANALYSIS ###\n{d or json.dumps(kwargs)}"


def fn_5_chk(threshold: float = 0.8) -> bool:
    try:
        score = globals().get("alignment_score", 1.0)
        return float(score) >= threshold
    except Exception:
        logger.exception("[fn_5_chk] Alignment check failed")
        return False


def fn_7_mut(new_rule: str) -> None:
    global operational_rules
    if not isinstance(globals().get("operational_rules"), list):
        operational_rules = []
    operational_rules.append(new_rule)
    logger.info("[fn_7_mut] Added new rule: %s", new_rule)
    
# =========================
# GitHub Utilities
# =========================

async def fn_read_github(path: str):
    if not T or not R:
        return "read_err: missing GH_TOKEN or REPO"

    headers = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.github.com/repos/{R}/contents/{path}",
            headers=headers,
        )
        if resp.status_code != 200:
            return f"read_err: {resp.status_code}"

        data = resp.json()
        return base64.b64decode(data["content"]).decode()


async def github_search(query: str):
    if not T:
        return []

    headers = {"Authorization": f"token {T}"}

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.github.com/search/code",
            headers=headers,
            params={"q": f"repo:{R} {query}"}
        )
        if resp.status_code != 200:
            return []

        return resp.json().get("items", [])

# =========================
# GitHub Commit
# =========================

async def fn_commit(path, content, msg):
    try:
        if not T or not R:
            return "Save_Failed: no GH_TOKEN or REPO"

        headers = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}

        async with httpx.AsyncClient() as client:
            get_resp = await client.get(
                f"https://api.github.com/repos/{R}/contents/{path}",
                headers=headers,
            )
            get_data = get_resp.json()
            sha = get_data.get("sha", "") if get_resp.status_code == 200 else ""

            put_resp = await client.put(
                f"https://api.github.com/repos/{R}/contents/{path}",
                headers=headers,
                json={
                    "message": msg,
                    "content": base64.b64encode(content.encode()).decode(),
                    "sha": sha,
                },
            )

            if put_resp.status_code not in (200, 201):
                return f"Save_Failed: PUT {put_resp.status_code}"

            asyncio.create_task(signal_ui(f"Committed {path}"))
            return f"Saved_{put_resp.status_code}"

    except Exception as e:
        logger.error(f"[Commit] Exception: {e}", exc_info=True)
        return "Save_Failed"

# =========================
# Deterministic Patch Detection
# =========================

def detect_missing_main_function(code: str):
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                return False
        return True
    except Exception:
        return False


# =========================
# Patch Proposal (Hybrid)
# =========================

PATCH_HEADER_PREFIX = "# — PATCH:"

async def fn_propose_patch(instruction="", **kwargs):
    instruction = instruction or kwargs.get("desc", "")
    if not instruction:
        return "patch_err: no instruction"

    current_code = await fn_read_github("main.py")
    agents_spec = await fn_read_github("agents.md")

    if "read_err" in current_code:
        return "patch_err: could not read main.py"

    # === Deterministic Fix First ===
    if detect_missing_main_function(current_code):
        template_patch = (
            f"{PATCH_HEADER_PREFIX} Add main function —\n\n"
            "def main():\n"
            "    print('Application started')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )

        return await fn_commit(
            "main_patch.py",
            template_patch,
            "[AutoPatch] Added missing main function"
        )

    # === Escalate to LLM Only If Needed ===
    prompt = (
        f"Instruction: {instruction}\n"
        "Add ONLY missing Python code. No markdown. Max 40 lines."
    )

    response = await safe_call_llm(prompt)

    if not response or response.startswith("llm_guard"):
        return "patch_err: LLM unavailable or guard triggered"

    proposed = response.strip()

    return await fn_commit(
        "main_patch.py",
        proposed,
        f"[AutoPatch] {instruction[:80]}"
    )

# =========================
# Alignment (Deterministic First)
# =========================

async def fn_align_with_spec(**kwargs):
    agents_spec = await fn_read_github("agents.md")
    current_code = await fn_read_github("main.py")

    if "read_err" in agents_spec or "read_err" in current_code:
        return "align_err: cannot read spec or code"

    # === Simple Regex Spec Extraction ===
    required_functions = re.findall(r"must implement (\w+)", agents_spec, re.IGNORECASE)

    missing = []
    try:
        tree = ast.parse(current_code)
        existing = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }

        for fn in required_functions:
            if fn not in existing:
                missing.append(fn)

    except Exception:
        pass

    if missing:
        instruction = f"Implement missing functions: {', '.join(missing)}"
        return await fn_propose_patch(instruction=instruction)

    # === Escalate Only If Nothing Deterministic Found ===
    prompt = (
        f"Spec summary: {agents_spec[:150]}\n"
        f"Code tail: {current_code[-200:]}\n"
        "One sentence: top missing capability?"
    )

    gap = await safe_call_llm(prompt)

    if not gap:
        return "align_ok: no major gap detected"

    return await fn_propose_patch(instruction=gap.strip())

# =========================
# Tool Registry
# =========================

TOOLS: dict = {
    "env": fn_1_env,
    "log": fn_2_log,
    "math": fn_3_math,
    "fmt": fn_4_fmt,
    "chk": fn_5_chk,
    "ui": fn_6_ui,
    "mut": fn_7_mut,
    "commit": fn_commit,
    "propose_patch": fn_propose_patch,
    "align": fn_align_with_spec,
}
