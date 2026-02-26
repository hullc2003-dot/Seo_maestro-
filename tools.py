“””
tools.py – Tool functions, GitHub helpers, prompt management, and tool/step registries.

Imports expected by main.py:
from tools import (
TOOLS, PRMPTS, STATE, signal_ui,
fn_commit, fn_reload_prompts, reload_prompts_from_agents_md,
SYSTEM_PROMPT_TEMPLATE, JSON_ENFORCEMENT,
)
“””

import asyncio
import ast
import base64
import json
import logging
import os
import subprocess
import sys

import httpx

logger = logging.getLogger(“AgentServer”)

# ─── SHARED STATE ─────────────────────────────────────────────────────────────

K = (os.getenv(“GROQ_API_KEY”) or “”).strip()
T = (os.getenv(“GH_TOKEN”)     or “”).strip()
R = (os.getenv(“REPO_PATH”)    or “”).strip()

STATE = {“rules”: “Goal: AI Engineer. Strategy: Deep Reflection over Speed.”, “lvl”: 1}

CTX_MAX_CHARS  = int(os.getenv(“CTX_MAX_CHARS”, 8000))
CONTACT_UI_URL = (os.getenv(“UI_STATUS_URL”, “”)).strip()

# ─── LLM PROMPT CONSTANTS ─────────────────────────────────────────────────────

JSON_ENFORCEMENT = (
“ Always respond with a single valid JSON object only – “
“no markdown, no prose, no code fences. “
‘Schema: {“tool”: “<n>”, “args”: {}, “thought”: “<reasoning>”}.’
)

SYSTEM_PROMPT_TEMPLATE = (
“{rules}. “
“You MUST respond with a single valid JSON object and nothing else – “
“no markdown, no prose, no code fences. “
“Your entire response must be parseable by json.loads(). “
’Schema: {“tool”: “<name>”, “args”: {}, “thought”: “<reasoning>”}. ’
“Valid tools: env(k), log(m), math(e), fmt(d), chk(g), ui(d), mut(p), “
“pip(package), lc(tool,input), read(path), propose_patch(instruction), “
“apply_patch(), align(), create_module(filename,code,description), “
“add_tool(name,code), add_step(prompt,position), list_tools(), list_steps(), “
“reload_prompts(), run_tests(filename), create_test(filename,code,description), “
“codegen(requirements). “
“Use exactly these argument names. Do not add extra fields.”
)

PATCH_HEADER_PREFIX = “# — PATCH:”

# ─── UI SIGNAL ────────────────────────────────────────────────────────────────

async def signal_ui(status: str):
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

# ─── GITHUB HELPERS ───────────────────────────────────────────────────────────

async def fn_commit(path, content, msg):
try:
if not T:
return “Save_Failed: no GH_TOKEN”
if not R:
return “Save_Failed: no REPO_PATH”
headers = {“Authorization”: f”token {T}”, “User-Agent”: “AIEngAgent”}
async with httpx.AsyncClient() as client:
get_resp = await client.get(
f”https://api.github.com/repos/{R}/contents/{path}”, headers=headers
)
get_data = get_resp.json()
if get_resp.status_code not in (200, 404):
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
if put_resp.status_code not in (200, 201):
return f”Save_Failed: PUT {put_resp.status_code}”
asyncio.create_task(signal_ui(f”Committed {path}”))
return f”Saved_{put_resp.status_code}”
except Exception as e:
logger.error(f”[Commit] Exception: {e}”, exc_info=True)
return “Save_Failed”

async def fn_read_github(path=””, **kwargs):
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
headers=headers,
timeout=15.0,
)
data = resp.json()
if resp.status_code != 200:
return f”read_err: {resp.status_code} {data.get(‘message’,’’)}”
content = base64.b64decode(data[“content”]).decode(“utf-8”, errors=“replace”)
return content[:6000]
except Exception as e:
return f”read_err: {e}”

# ─── PACKAGE BOOTSTRAP ────────────────────────────────────────────────────────

def _ensure_pkg(pkg, import_as=None):
name = import_as or pkg.split(”[”)[0]
try:
**import**(name)
except ImportError:
logger.info(f”[Bootstrap] Installing {pkg}…”)
subprocess.check_call([sys.executable, “-m”, “pip”, “install”, pkg, “–quiet”])

# ─── CORE TOOLS ───────────────────────────────────────────────────────────────

def fn_1_env(k=””, **kwargs):
return os.getenv(k, “Null”)

def fn_2_log(m=None, **kwargs):
msg = m or json.dumps(kwargs) or “Log recorded”
logger.info(f”[Reflect]: {msg}”)
return “Log recorded”

def fn_3_math(e=None, expression=None, expr=None, **kwargs):
formula = e or expression or expr or “”
if not formula:
return “Math Err: no expression”
try:
from simpleeval import simple_eval
return simple_eval(formula)
except Exception:
return “Math Err”

def fn_4_fmt(d=””, **kwargs):
return f”### ANALYSIS ###\n{d or json.dumps(kwargs)}”

def fn_5_chk(threshold: float = 0.8) -> bool:
“”“Validate current alignment against the given threshold.”””
try:
score = globals().get(“alignment_score”, 1.0)
return float(score) >= threshold
except Exception:
logger.exception(”[fn_5_chk] Alignment check failed”)
return False

def fn_6_ui(d=””, **kwargs):
return f”UI_UPDATE: {d or json.dumps(kwargs)}”

def fn_7_mut(new_rule: str) -> None:
“”“Mutate the agent’s operational rules when alignment is insufficient.”””
global operational_rules
if not isinstance(operational_rules, list):
operational_rules = []
operational_rules.append(new_rule)
logger.info(”[fn_7_mut] Added new rule: %s”, new_rule)
asyncio.create_task(signal_ui(“Operational rules mutated”))

def fn_8_pip(package=””, **kwargs):
pkg = package or kwargs.get(“pkg”, “”)
if not pkg:
return “pip_err: no package name”
try:
_ensure_pkg(pkg)
return f”Installed: {pkg}”
except Exception as e:
return f”pip_err: {e}”

def fn_9_lc_tool(tool=””, input=””, **kwargs):
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
return f”lc_err: {e}”

# ─── PATCH TOOLS ──────────────────────────────────────────────────────────────

async def fn_propose_patch(instruction=””, **kwargs):
instruction = instruction or kwargs.get(“desc”, “”)
if not instruction:
return “patch_err: no instruction”
current_code = await fn_read_github(“main.py”)
agents_spec  = await fn_read_github(“agents.md”)
if “read_err” in current_code:
return “patch_err: could not read main.py”
if “read_err” in agents_spec:
return “patch_err: could not read agents.md”
spec_head = agents_spec[:250].replace(”\n”, “ “)
code_tail = current_code[-350:]
prompt = (
f”agents.md goal: {spec_head}\n\nmain.py tail:\n{code_tail}\n\nTask: {instruction}\n\n”
“Add ONLY the missing Python. No duplicate imports. “
f”First line MUST be: {PATCH_HEADER_PREFIX} <desc> —\nNo markdown. Raw Python only. Max 40 lines.”
)
try:
async with httpx.AsyncClient() as client:
resp = await client.post(
“https://api.groq.com/openai/v1/chat/completions”,
headers={“Authorization”: f”Bearer {K}”, “Content-Type”: “application/json”},
json={
“model”: “groq/compound”,
“messages”: [
{“role”: “system”, “content”: “You are a Python code generator. Output raw Python only.”},
{“role”: “user”,   “content”: prompt},
],
“max_tokens”: 1024,
},
timeout=60.0,
)
rdata2 = resp.json()
if “choices” not in rdata2:
return f”patch_err: groq={rdata2.get(‘error’, rdata2)}”
proposed = rdata2[“choices”][0][“message”][“content”].strip()
if proposed.startswith(”`"): proposed = proposed.split("\n", 1)[1].rsplit("`”, 1)[0]
result = await fn_commit(“main_patch.py”, proposed, f”[AutoPatch] {instruction[:80]}”)
return f”Proposed patch committed as main_patch.py — {result}”
except Exception as e:
return f”patch_err: {e}”

async def fn_apply_patch(**kwargs):
patch = await fn_read_github(“main_patch.py”)
if “read_err” in patch:
return f”apply_err: {patch}”
if PATCH_HEADER_PREFIX not in patch:
return f”apply_err: patch missing header ‘{PATCH_HEADER_PREFIX}’”
current = await fn_read_github(“main.py”)
if “read_err” in current:
return “apply_err: could not read main.py”
INSERT_MARKER = “# — CATCH-ALL POST”
if INSERT_MARKER in current:
updated = current.replace(INSERT_MARKER, patch.strip() + “\n\n\n” + INSERT_MARKER, 1)
else:
updated = current + “\n\n” + patch.strip()
if “FastAPI” not in updated or “run_autonomous_loop” not in updated:
return “apply_err: merged file failed sanity check”
result = await fn_commit(“main.py”, updated, “[AutoApply] Append patch to main.py”)
return f”main.py patched — {result}”

async def fn_align_with_spec(**kwargs):
agents_spec = await fn_read_github(“agents.md”)
if “read_err” in agents_spec:
return f”align_skipped: {agents_spec}”
current_code = await fn_read_github(“main.py”)
spec_snippet = agents_spec[:150].replace(”\n”, “ “)
code_snippet = current_code[-200:].replace(”\n”, “ “)
gap_prompt = (
f”spec: {spec_snippet} | “
f”code tail: {code_snippet} | “
“ONE sentence: top missing capability?”
)
gap_prompt = gap_prompt[:600]
try:
await asyncio.sleep(8)
async with httpx.AsyncClient() as client:
resp = await client.post(
“https://api.groq.com/openai/v1/chat/completions”,
headers={“Authorization”: f”Bearer {K}”, “Content-Type”: “application/json”},
json={
“model”: “compound-beta”,
“messages”: [{“role”: “user”, “content”: gap_prompt}],
“max_tokens”: 200,
},
timeout=30.0,
)
rdata = resp.json()
if “choices” not in rdata:
return f”align_partial: {rdata.get(‘error’, rdata)}”
gap = rdata[“choices”][0][“message”][“content”].strip()
logger.info(f”[Align] Gap: {gap}”)
await asyncio.sleep(10)
patch_result = await fn_propose_patch(instruction=gap)
return f”Gap: {gap} | {patch_result}”
except Exception as e:
return f”align_err: {e}”

# ─── MODULE / TEST TOOLS ──────────────────────────────────────────────────────

async def fn_create_module(filename=””, code=””, description=””, **kwargs):
“”“Create a brand-new Python module in the GitHub repo.”””
filename    = filename    or kwargs.get(“file”, “”)
code        = code        or kwargs.get(“source”, “”) or kwargs.get(“content”, “”)
description = description or kwargs.get(“desc”, “New module”)
if not filename:
return “module_err: no filename provided”
if not code or not code.strip():
return “module_err: no code provided”
if not filename.endswith(”.py”):
filename += “.py”
if “..” in filename or filename.startswith(”/”):
return “module_err: unsafe filename – no ‘..’ or leading ‘/’”
try:
ast.parse(code)
except SyntaxError as se:
return f”module_err: syntax error in provided code – {se}”
commit_msg = f”[AutoModule] {description[:80]} → {filename}”
result = await fn_commit(filename, code, commit_msg)
logger.info(f”[CreateModule] {filename} → {result}”)
asyncio.create_task(signal_ui(f”Module created: {filename}”))
return f”module_created: {filename} | commit={result}”

async def fn_create_test(filename=””, code=””, description=””, **kwargs):
“”“Create a pytest-compatible test module in the GitHub repo.”””
import os.path as *op
filename    = filename    or kwargs.get(“file”, “”)
code        = code        or kwargs.get(“source”, “”) or kwargs.get(“content”, “”)
description = description or kwargs.get(“desc”, “New test module”)
if not filename:
return “test_err: no filename”
if not code or not code.strip():
return “test_err: no code”
if not filename.endswith(”.py”):
filename += “.py”
if “..” in filename or filename.startswith(”/”):
return “test_err: unsafe filename”
dirname, basename = *op.split(filename)
if not basename.startswith(“test*”):
basename = “test*” + basename
filename = _op.join(dirname, basename).lstrip(”/”)
try:
ast.parse(code)
except SyntaxError as se:
return f”test_err: syntax error - {se}”
result = await fn_commit(filename, code, f”[AutoTest] {description[:80]} -> {filename}”)
logger.info(f”[CreateTest] {filename} -> {result}”)
asyncio.create_task(signal_ui(f”Test created: {filename}”))
return f”test_created: {filename} | commit={result}”

async def fn_run_tests(filename=””, **kwargs):
“”“Download a test file (or all tests/) from the repo and run with pytest.”””
import tempfile
import subprocess as _sp
import os as _os

```
target = filename or kwargs.get("file", "")
if target:
    files_to_fetch = [target]
else:
    if not T or not R:
        return "test_run_err: missing GH_TOKEN or REPO_PATH"
    try:
        headers = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"https://api.github.com/repos/{R}/contents/tests",
                headers=headers,
                timeout=15.0,
            )
        if resp.status_code != 200:
            return f"test_run_err: HTTP {resp.status_code} listing tests/"
        files_to_fetch = [
            item["path"]
            for item in resp.json()
            if item.get("type") == "file"
            and item["name"].startswith("test_")
            and item["name"].endswith(".py")
        ]
        if not files_to_fetch:
            return "test_run_err: no test_*.py files in tests/"
    except Exception as exc:
        return f"test_run_err: {exc}"

results: list[str] = []
with tempfile.TemporaryDirectory(prefix="agent_tests_") as tmpdir:
    for fpath in files_to_fetch:
        basename = _os.path.basename(fpath)
        if not basename.startswith("test_"):
            results.append(f"SKIP {fpath}: must start with test_")
            continue
        src = await fn_read_github(fpath)
        if "read_err" in src:
            results.append(f"FETCH_ERR {fpath}: {src}")
            continue
        try:
            ast.parse(src)
        except SyntaxError as se:
            results.append(f"SYNTAX_ERR {fpath}: {se}")
            continue
        local_path = _os.path.join(tmpdir, basename)
        with open(local_path, "w", encoding="utf-8") as fh:
            fh.write(src)
        try:
            proc = _sp.run(
                [sys.executable, "-m", "pytest", local_path,
                 "-v", "--tb=short", "--no-header", "-q"],
                capture_output=True, text=True, timeout=60, cwd=tmpdir,
            )
            out_lines = (proc.stdout + proc.stderr).strip().splitlines()
            condensed = "\n".join(out_lines[-40:])
            status = "PASS" if proc.returncode == 0 else "FAIL"
            results.append(f"{status} [{fpath}]\n{condensed}")
        except _sp.TimeoutExpired:
            results.append(f"TIMEOUT [{fpath}]: exceeded 60s")
        except Exception as exc:
            results.append(f"RUN_ERR [{fpath}]: {exc}")

summary = "\n\n".join(results)
logger.info(f"[RunTests] {len(files_to_fetch)} file(s) run")
asyncio.create_task(signal_ui(f"Tests done: {len(results)} file(s)"))
return summary[:2000]
```

# ─── DYNAMIC TOOL / STEP REGISTRATION ────────────────────────────────────────

def fn_add_tool(name=””, code=””, **kwargs):
“”“Define and register a new callable tool at runtime without restarting.”””
import keyword as _kw

```
name = name or kwargs.get("tool_name", "")
code = code or kwargs.get("source", "")
if not name or not name.strip():
    return "add_tool_err: no tool name provided"
if not code or not code.strip():
    return "add_tool_err: no code provided"
if name in TOOLS:
    return f"add_tool_err: '{name}' already exists – pick a different name"
if not name.isidentifier() or _kw.iskeyword(name):
    return f"add_tool_err: '{name}' is not a valid Python identifier"
try:
    ast.parse(code)
except SyntaxError as se:
    return f"add_tool_err: syntax error – {se}"
ns: dict = {}
try:
    exec(compile(code, f"<tool:{name}>", "exec"), ns)
except Exception as exc:
    return f"add_tool_err: exec failed – {exc}"
fn = ns.get(name)
if fn is None:
    return f"add_tool_err: code did not define a function named '{name}'"
if not callable(fn):
    return f"add_tool_err: '{name}' is not callable"
TOOLS[name] = fn
global SYSTEM_PROMPT_TEMPLATE
if name not in SYSTEM_PROMPT_TEMPLATE:
    SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE.rstrip(". ") + f", {name}()."
logger.info(f"[AddTool] Registered: '{name}'")
return f"tool_registered: {name} | total_tools={len(TOOLS)}"
```

def fn_add_step(prompt=””, position=None, **kwargs):
“”“Inject a new directive step into PRMPTS at runtime.”””
prompt = prompt or kwargs.get(“directive”, “”) or kwargs.get(“p”, “”)
if not prompt or not prompt.strip():
return “add_step_err: no prompt text provided”
pos = position if position is not None else kwargs.get(“pos”)
if pos is None:
pos = len(PRMPTS) - 1
try:
pos = int(pos)
except (TypeError, ValueError):
pos = len(PRMPTS) - 1
pos = max(0, min(pos, len(PRMPTS) - 1))
PRMPTS.insert(pos, prompt.strip())
logger.info(f”[AddStep] Inserted step {pos}: {prompt[:80]}”)
return f”step_added: index={pos} | total_steps={len(PRMPTS)}”

def fn_list_tools(**kwargs):
“”“Return a JSON-serialisable list of all currently registered tool names.”””
return {“tools”: sorted(TOOLS.keys()), “count”: len(TOOLS)}

def fn_list_steps(**kwargs):
“”“Return all current PRMPTS with their index.”””
return {
“steps”: [{“index”: i, “prompt”: p[:120]} for i, p in enumerate(PRMPTS)],
“total”: len(PRMPTS),
}

async def fn_reload_prompts(**kwargs):
“”“Tool: reload PRMPTS from agents.md on demand.”””
updated = await reload_prompts_from_agents_md()
return f”prompts_reloaded: {len(PRMPTS)} steps | from_agents_md={updated}”

# ─── PROMPT MANAGEMENT ────────────────────────────────────────────────────────

_DEFAULT_PRMPTS: list[str] = [
“Critically analyze the current state. What is missing to reach AI Engineer status? You MUST call tool=‘chk’ with args={‘g’: ‘<your one-sentence gap summary>’}.”,
“Generate a hypothesis for a better autonomous pattern. You MUST call tool=‘log’ with args={‘m’: ‘<your hypothesis in one sentence>’}.”,
“Identify the single most critical failure point in the previous step. You MUST call tool=‘fmt’ with args={‘d’: ‘<failure point in one sentence>’}.”,
“You MUST call tool=‘mut’ with args={‘p’: ‘<one sentence ruleset>’}. Keep p under 100 characters. No markdown.”,
“Log a one-sentence final verification summary. You MUST call tool=‘log’ with args={‘m’: ‘<verification summary>’}.”,
“MANDATORY FINAL STEP - no other tool is valid here. You MUST call tool=‘align’ with args={}. Do NOT call chk, log, or any other tool.”,
]

PRMPTS: list[str] = list(_DEFAULT_PRMPTS)

def _parse_prompts_from_md(md: str) -> list[str]:
“”“Parse prompts from a ## Prompts fenced block in agents.md.”””
import re as _re
section = _re.search(
r’##\s*Prompts.*?`(?:\w*)\n(.*?)`’,
md, _re.IGNORECASE | _re.DOTALL
)
if not section:
return []
prompts: list[str] = []
for line in section.group(1).splitlines():
line = line.strip()
if not line:
continue
line = _re.sub(r’^(?:\d+.|-|*)\s*’, ‘’, line).strip()
if line:
prompts.append(line)
return prompts

async def reload_prompts_from_agents_md() -> bool:
“”“Fetch agents.md and update PRMPTS from its ## Prompts block.”””
global PRMPTS
ALIGN_DIRECTIVE = (
“MANDATORY FINAL STEP - no other tool is valid here. “
“You MUST call tool=‘align’ with args={}. “
“Do NOT call chk, log, or any other tool.”
)
md = await fn_read_github(“agents.md”)
if “read_err” in md:
logger.warning(f”[Prompts] Could not read agents.md: {md}”)
return False
parsed = _parse_prompts_from_md(md)
if not parsed:
logger.info(”[Prompts] No ## Prompts block in agents.md - keeping current prompts”)
return False
align_steps = [p for p in parsed if “align” in p.lower()]
other_steps = [p for p in parsed if “align” not in p.lower()]
if not align_steps:
align_steps = [ALIGN_DIRECTIVE]
PRMPTS = other_steps + align_steps
logger.info(f”[Prompts] Loaded {len(PRMPTS)} steps from agents.md”)
return True

# ─── AUTONOMOUS CODE DEVELOPMENT ─────────────────────────────────────────────

def _render_template(req: dict) -> str:
“”“Create a simple function based on a high-level requirement.”””
name      = req.get(“func”, “generated_func”)
params    = req.get(“params”, [])
body      = req.get(“body”, “return None”)
param_str = “, “.join(params)
return f”def {name}({param_str}):\n    {body}\n”

def generate_code(requirements: dict) -> str:
“”“Generate Python source code from a requirement dict.”””
return _render_template(requirements)

def _load_function(code: str, func_name: str):
“”“Dynamically load a generated function from source.”””
namespace: dict = {}
exec(code, namespace)
return namespace[func_name]

def test_code(code: str, requirements: dict) -> bool:
“”“Run supplied test cases against the generated function.”””
tests = requirements.get(“tests”, [])
if not tests:
return True
func_name = requirements.get(“func”, “generated_func”)
try:
func = _load_function(code, func_name)
except Exception:
return False
for case in tests:
args     = case.get(“args”, [])
expected = case.get(“expected”)
try:
result = func(*args)
except Exception:
return False
if result != expected:
return False
return True

def validate_code(code: str, requirements: dict) -> bool:
“”“Validate that generated code meets structural requirements.”””
required_name = requirements.get(“func”)
if required_name and f”def {required_name}(” not in code:
return False
return True

async def autonomous_code_dev(requirements: dict = None, **kwargs) -> str:
“””
Generate, test, and validate Python code autonomously from a requirement dict.

```
Tool args the LLM should send:
    requirements – dict with keys:
        func    (str)        – function name
        params  (list[str])  – parameter names
        body    (str)        – single-line function body, e.g. 'return a + b'
        tests   (list[dict]) – each dict has 'args' (list) and 'expected' (any)
"""
req = requirements or kwargs.get("req") or kwargs
if not req:
    return "codegen_err: no requirements provided"
code = generate_code(req)
if not test_code(code, req):
    return "codegen_err: test cases failed"
if not validate_code(code, req):
    return "codegen_err: validation failed"
logger.info(f"[CodeDev] Generated function '{req.get('func', '?')}' successfully")
return code
```

# ─── TOOL REGISTRY ────────────────────────────────────────────────────────────

TOOLS: dict = {
“env”:            fn_1_env,
“log”:            fn_2_log,
“math”:           fn_3_math,
“fmt”:            fn_4_fmt,
“chk”:            fn_5_chk,
“ui”:             fn_6_ui,
“mut”:            fn_7_mut,
“commit”:         fn_commit,
“pip”:            fn_8_pip,
“lc”:             fn_9_lc_tool,
“read”:           fn_read_github,
“propose_patch”:  fn_propose_patch,
“apply_patch”:    fn_apply_patch,
“align”:          fn_align_with_spec,
“create_module”:  fn_create_module,
“add_tool”:       fn_add_tool,
“add_step”:       fn_add_step,
“list_tools”:     fn_list_tools,
“list_steps”:     fn_list_steps,
“reload_prompts”: fn_reload_prompts,
“create_test”:    fn_create_test,
“run_tests”:      fn_run_tests,
“codegen”:        autonomous_code_dev,
}
