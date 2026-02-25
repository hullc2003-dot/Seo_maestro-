# module2.py: Tools and Functions (OpenRouter Integrated)

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
import tempfile
from typing import Dict, Any, Callable

from module1 import T, R, STATE, CTX_MAX_CHARS, signal_ui, logger, call_llm

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


def fn_6_ui(d="", **kwargs):
    return f"UI_UPDATE: {d or json.dumps(kwargs)}"


def fn_7_mut(new_rule: str) -> None:
    global operational_rules
    if not isinstance(globals().get("operational_rules"), list):
        operational_rules = []
    operational_rules.append(new_rule)
    logger.info("[fn_7_mut] Added new rule: %s", new_rule)
    asyncio.create_task(signal_ui("Operational rules mutated"))

# =========================
# GitHub Commit
# =========================

async def fn_commit(path, content, msg):
    try:
        if not T:
            return "Save_Failed: no GH_TOKEN"
        if not R:
            return "Save_Failed: no REPO_PATH"

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
# LLM Patch Proposal (OpenRouter)
# =========================

PATCH_HEADER_PREFIX = "# — PATCH:"

async def fn_propose_patch(instruction="", **kwargs):
    instruction = instruction or kwargs.get("desc", "")
    if not instruction:
        return "patch_err: no instruction"

    current_code = await fn_read_github("main.py")
    agents_spec  = await fn_read_github("agents.md")

    if "read_err" in current_code:
        return "patch_err: could not read main.py"
    if "read_err" in agents_spec:
        return "patch_err: could not read agents.md"

    spec_head = agents_spec[:250].replace("\n", " ")
    code_tail = current_code[-350:]

    prompt = (
        f"agents.md goal: {spec_head}\n\n"
        f"main.py tail:\n{code_tail}\n\n"
        f"Task: {instruction}\n\n"
        "Add ONLY the missing Python. No duplicate imports. "
        f"First line MUST be: {PATCH_HEADER_PREFIX} <desc> —\n"
        "No markdown. Raw Python only. Max 40 lines."
    )

    try:
        response = await call_llm(prompt)

        if not response:
            return "patch_err: empty LLM response"

        proposed = response.strip()

        result = await fn_commit(
            "main_patch.py",
            proposed,
            f"[AutoPatch] {instruction[:80]}"
        )

        return f"Proposed patch committed as main_patch.py — {result}"

    except Exception as e:
        return f"patch_err: {e}"

# =========================
# Alignment (OpenRouter)
# =========================

async def fn_align_with_spec(**kwargs):
    agents_spec = await fn_read_github("agents.md")
    current_code = await fn_read_github("main.py")

    if "read_err" in agents_spec or "read_err" in current_code:
        return "align_err: cannot read spec or code"

    spec_snippet = agents_spec[:150].replace("\n", " ")
    code_snippet = current_code[-200:].replace("\n", " ")

    gap_prompt = (
        f"spec: {spec_snippet} | "
        f"code tail: {code_snippet} | "
        "ONE sentence: top missing capability?"
    )

    try:
        gap = await call_llm(gap_prompt)
        gap = gap.strip()

        logger.info(f"[Align] Gap: {gap}")

        patch_result = await fn_propose_patch(instruction=gap)

        return f"Gap: {gap} | {patch_result}"

    except Exception as e:
        return f"align_err: {e}"

# =========================
# Tool Registry
# =========================

TOOLS: dict = {
    "env":            fn_1_env,
    "log":            fn_2_log,
    "math":           fn_3_math,
    "fmt":            fn_4_fmt,
    "chk":            fn_5_chk,
    "ui":             fn_6_ui,
    "mut":            fn_7_mut,
    "commit":         fn_commit,
    "propose_patch":  fn_propose_patch,
    "align":          fn_align_with_spec,
}
