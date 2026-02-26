# module3.py: Autonomous Loop and Prompts

import asyncio
import json
import re
from module1 import CTX_MAX_CHARS, call_llm, logger
from module2 import TOOLS, fn_commit, fn_read_github

_DEFAULT_PRMPTS = [
    "Critically analyze the current state. What is missing to reach AI Engineer status? You MUST call tool='chk' with args={'g': '<your one-sentence gap summary>'}.",
    "Generate a hypothesis for a better autonomous pattern. You MUST call tool='log' with args={'m': '<your hypothesis in one sentence>'}.",
    "Identify the single most critical failure point in the previous step. You MUST call tool='fmt' with args={'d': '<failure point in one sentence>'}.",
    "You MUST call tool='mut' with args={'p': '<one sentence ruleset>'}. Keep p under 100 characters. No markdown.",
    "Log a one-sentence final verification summary. You MUST call tool='log' with args={'m': '<verification summary>'}.",
    "MANDATORY FINAL STEP - no other tool is valid here. You MUST call tool='align' with args={}. Do NOT call chk, log, or any other tool.",
]

PRMPTS = list(_DEFAULT_PRMPTS)

async def reload_prompts_from_agents_md():
    # Placeholder: in production, dynamically reload prompts from agents.md
    pass

async def run_autonomous_loop(input_str: str) -> str:
    ctx = input_str
    await reload_prompts_from_agents_md()
    logger.info(f"[Loop] Starting with {len(PRMPTS)} steps")

    i = 0
    while i < len(PRMPTS):
        ctx_payload = ctx[-CTX_MAX_CHARS:] if len(ctx) > CTX_MAX_CHARS else ctx
        ctx_snip = ctx_payload[-1200:]
        directive = PRMPTS[i][:400]
        user_msg = f"STEP {i}. Context: {ctx_snip}\nDirective: {directive}"
        if len(user_msg) > 2800:
            user_msg = user_msg[-2800:]

        raw = await call_llm(user_msg)
        if not raw:
            logger.warning(f"[Loop] Step {i}: empty LLM response, skipping")
            i += 1
            continue

        # Attempt to parse JSON from LLM response
        try:
            data = json.loads(raw)
        except Exception:
            match = re.search(r'[{].*[}]', raw, re.DOTALL)
            data = None
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    pass
            if not data:
                mut_idx = 3
                if i == mut_idx:
                    safe = raw[:500].replace('"', "'").replace('\n', ' ').strip()
                    data = {"tool": "mut", "args": {"p": safe}, "thought": "extracted from markdown"}
                else:
                    logger.error(f"[Loop] Step {i}: parse failed | raw={raw!r}")
                    i += 1
                    continue

        t, a = data.get("tool"), data.get("args", {})
        align_idx = len(PRMPTS) - 1
        mut_idx = 3

        # Force specific tools if LLM misbehaves
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

        # Call the tool
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
    return ctx
