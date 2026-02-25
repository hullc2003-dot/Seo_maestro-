# module3.py: Autonomous Loop (Optimized + Guarded)

import asyncio
import json
import time
import re
import hashlib

from module1 import CTX_MAX_CHARS, call_llm, logger, signal_ui
from module2 import TOOLS, fn_commit, reload_prompts_from_agents_md

# =========================
# Loop Guardrails
# =========================

LOOP_LLM_LIMIT = 6
loop_llm_count = 0

async def safe_loop_llm(prompt: str) -> str:
    global loop_llm_count

    if loop_llm_count >= LOOP_LLM_LIMIT:
        logger.warning("[Loop] LLM limit reached — skipping call")
        return ""

    loop_llm_count += 1
    return await call_llm(prompt)


# =========================
# Default Prompts
# =========================

_DEFAULT_PRMPTS: list[str] = [
    "Critically analyze the current state. You MUST call tool='chk' with args={'g': '<one-sentence gap>'}.",
    "Generate a better autonomous hypothesis. You MUST call tool='log' with args={'m': '<one sentence>'}.",
    "Identify the single critical failure point. You MUST call tool='fmt' with args={'d': '<one sentence>'}.",
    "You MUST call tool='mut' with args={'p': '<ruleset under 100 chars>'}. No markdown.",
    "Log a one-sentence final verification summary. You MUST call tool='log' with args={'m': '<verification>'}.",
    "MANDATORY FINAL STEP. You MUST call tool='align' with args={}.",
]

PRMPTS: list[str] = list(_DEFAULT_PRMPTS)


# =========================
# JSON Extraction Helper
# =========================

def extract_json(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                return None
        return None


# =========================
# Autonomous Loop
# =========================

async def run_autonomous_loop(input_str: str) -> str:
    global loop_llm_count
    loop_llm_count = 0

    ctx = input_str
    await reload_prompts_from_agents_md()
    logger.info(f"[Loop] Starting with {len(PRMPTS)} steps")

    for i, directive in enumerate(PRMPTS):

        # === Smart Context Shrinking ===
        ctx_payload = ctx[-CTX_MAX_CHARS:]
        ctx_snip = ctx_payload[-1000:]

        user_msg = f"STEP {i}\nContext:\n{ctx_snip}\nDirective:\n{directive}"
        user_msg = user_msg[-2600:]

        raw = await safe_loop_llm(user_msg)

        if not raw:
            logger.warning(f"[Loop] Step {i}: no LLM output")
            continue

        data = extract_json(raw)

        # === Auto-repair for mut step if markdown ===
        if not data and i == 3:
            safe = raw[:400].replace('"', "'").replace("\n", " ").strip()
            data = {"tool": "mut", "args": {"p": safe}}

        if not data:
            logger.warning(f"[Loop] Step {i}: JSON parse failed")
            continue

        tool_name = data.get("tool")
        args = data.get("args", {})
        thought = data.get("thought", "")

        # === Hard Constraints ===
        if i == len(PRMPTS) - 1:
            tool_name = "align"
            args = {}

        if i == 4 and tool_name != "log":
            tool_name = "log"
            args = {"m": f"Verification complete. {ctx[-100:].replace(chr(10),' ')}"}

        if tool_name == "commit":
            logger.warning("[Loop] Mid-loop commit blocked")
            continue

        # === Tool Execution ===
        if tool_name in TOOLS:
            try:
                result = TOOLS[tool_name](**args)
                if asyncio.iscoroutine(result):
                    result = await result
            except Exception as exc:
                result = f"Tool error: {exc}"
                logger.error(f"[Loop] Tool '{tool_name}' error: {exc}", exc_info=True)

            ctx += f"\n[Step {i}] Tool: {tool_name} | Result: {result} | Thought: {thought}"

        else:
            ctx += f"\n[Step {i}] Unknown tool '{tool_name}' | Thought: {thought}"

    # === Final Commit ===
    commit_result = await fn_commit(
        "engineer_log.md",
        ctx,
        "Intellectual Evolution Log"
    )

    logger.info(f"[Commit] {commit_result}")
    await signal_ui("Loop complete")

    return ctx
