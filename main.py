import os, json, base64, time, asyncio, logging, subprocess, sys
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

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

K = (os.getenv("GROQ_API_KEY") or "").strip()
T = (os.getenv("GH_TOKEN")     or "").strip()
R = (os.getenv("REPO_PATH")    or "").strip()
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

def fn_1_env(k="", **kwargs): return os.getenv(k, "Null")

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

def fn_4_fmt(d="", **kwargs): return f"### ANALYSIS ###\n{d or json.dumps(kwargs)}"

def fn_5_chk(threshold: float = 0.8) -> bool:
    """
    Validate current alignment against the given threshold.
    Returns True if alignment >= threshold, else False.
    """
    try:
        # `alignment_score` should be maintained elsewhere in the agent.
        score = globals().get("alignment_score", 1.0)
        return float(score) >= threshold
    except Exception:
        logger.exception("[fn_5_chk] Alignment check failed")
        return False

def fn_6_ui(d="", **kwargs): return f"UI_UPDATE: {d or json.dumps(kwargs)}"

def fn_7_mut(new_rule: str) -> None:
    """
    Mutate the agent's operational rules when alignment is insufficient.
    Appends `new_rule` to the global `operational_rules` list and notifies the UI.
    """
    global operational_rules
    if not isinstance(operational_rules, list):
        operational_rules = []
    operational_rules.append(new_rule)
    logger.info("[fn_7_mut] Added new rule: %s", new_rule)
    # Signal the UI that rules have been mutated
    asyncio.create_task(signal_ui("Operational rules mutated"))

JSON_ENFORCEMENT = (
    " Always respond with a single valid JSON object only – "
    "no markdown, no prose, no code fences. "
    'Schema: {"tool": "<n>", "args": {}, "thought": "<reasoning>"}.'
)

async def fn_commit(path, content, msg):
    try:
        if not T:
            return "Save_Failed: no GH_TOKEN"
        if not R:
            return "Save_Failed: no REPO_PATH"
        headers = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}
        async with httpx.AsyncClient() as client:
            get_resp = await client.get(f"https://api.github.com/repos/{R}/contents/{path}", headers=headers)
            get_data = get_resp.json()
            if get_resp.status_code not in (200, 404):
                return f"Save_Failed: GET {get_resp.status_code}"
            sha = get_data.get("sha", "")
            put_resp = await client.put(
                f"https://api.github.com/repos/{R}/contents/{path}",
                headers=headers,
                json={"message": msg, "content": base64.b64encode(content.encode()).decode(), "sha": sha},
            )
            if put_resp.status_code not in (200, 201):
                return f"Save_Failed: PUT {put_resp.status_code}"
            asyncio.create_task(signal_ui(f"Committed {path}"))
            return f"Saved_{put_resp.status_code}"
    except Exception as e:
        logger.error(f"[Commit] Exception: {e}", exc_info=True)
        return "Save_Failed"

def _ensure_pkg(pkg, import_as=None):
    name = import_as or pkg.split("[")[0]
    try:
        __import__(name)
    except ImportError:
        logger.info(f"[Bootstrap] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

def fn_8_pip(package="", **kwargs):
    pkg = package or kwargs.get("pkg", "")
    if not pkg:
        return "pip_err: no package name"
    try:
        _ensure_pkg(pkg)
        return f"Installed: {pkg}"
    except Exception as e:
        return f"pip_err: {e}"

def fn_9_lc_tool(tool="", input="", **kwargs):
    tool = tool or kwargs.get("name", "")
    inp  = input or kwargs.get("query", "") or kwargs.get("input", "")
    if not tool or not inp:
        return "lc_err: need tool and input"
    try:
        _ensure_pkg("langchain-community", "langchain_community")
        _ensure_pkg("duckduckgo-search",   "duckduckgo_search")
        if tool in ("ddg-search", "search", "web"):
            from langchain_community.tools import DuckDuckGoSearchRun
            return DuckDuckGoSearchRun().run(inp)[:1500]
        if tool in ("wikipedia", "wiki"):
            _ensure_pkg("wikipedia", "wikipedia")
            from langchain_community.tools import WikipediaQueryRun
            from langchain_community.utilities import WikipediaAPIWrapper
            return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run(inp)[:1500]
        return f"lc_err: unknown tool '{tool}'"
    except Exception as e:
        return f"lc_err: {e}"

async def fn_read_github(path="", **kwargs):
    path = path or kwargs.get("file", "")
    if not path:
        return "read_err: no path"
    if not T or not R:
        return "read_err: missing GH_TOKEN or REPO_PATH"
    try:
        headers = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://api.github.com/repos/{R}/contents/{path}", headers=headers, timeout=15.0)
            data = resp.json()
            if resp.status_code != 200:
                return f"read_err: {resp.status_code} {data.get('message','')}"
            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            return content[:6000]
    except Exception as e:
        return f"read_err: {e}"

PATCH_HEADER_PREFIX = "# — PATCH:"

async def fn_propose_patch(instruction="", **kwargs):
    instruction = instruction or kwargs.get("desc", "")
    if not instruction:
        return "patch_err: no instruction"
    current_code = await fn_read_github("main.py")
    agents_spec  = await fn_read_github("agents.md")
    if "read_err" in current_code:
        return f"patch_err: could not read main.py"
    if "read_err" in agents_spec:
        return f"patch_err: could not read agents.md"
    spec_head = agents_spec[:250].replace("\n", " ")
    code_tail = current_code[-350:]
    prompt = (
        f"agents.md goal: {spec_head}\n\nmain.py tail:\n{code_tail}\n\nTask: {instruction}\n\n"
        "Add ONLY the missing Python. No duplicate imports. "
        f"First line MUST be: {PATCH_HEADER_PREFIX} <desc> —\nNo markdown. Raw Python only. Max 40 lines."
    )
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {K}", "Content-Type": "application/json"},
                json={"model": "groq/compound", "messages": [
                    {"role": "system", "content": "You are a Python code generator. Output raw Python only."},
                    {"role": "user",   "content": prompt},
                ], "max_tokens": 1024},
                timeout=60.0,
            )
            rdata2 = resp.json()
            if "choices" not in rdata2:
                return f"patch_err: groq={rdata2.get('error', rdata2)}"
            proposed = rdata2["choices"][0]["message"]["content"].strip()
            if proposed.startswith("```"): proposed = proposed.split("\n", 1)[1].rsplit("```", 1)[0]
            result = await fn_commit("main_patch.py", proposed, f"[AutoPatch] {instruction[:80]}")
            return f"Proposed patch committed as main_patch.py — {result}"
    except Exception as e:
        return f"patch_err: {e}"

async def fn_apply_patch(**kwargs):
    patch = await fn_read_github("main_patch.py")
    if "read_err" in patch:
        return f"apply_err: {patch}"
    if PATCH_HEADER_PREFIX not in patch:
        return f"apply_err: patch missing header '{PATCH_HEADER_PREFIX}'"
    current = await fn_read_github("main.py")
    if "read_err" in current:
        return f"apply_err: could not read main.py"
    INSERT_MARKER = "# — CATCH-ALL POST"
    if INSERT_MARKER in current:
        updated = current.replace(INSERT_MARKER, patch.strip() + "\n\n\n" + INSERT_MARKER, 1)
    else:
        updated = current + "\n\n" + patch.strip()
    if "FastAPI" not in updated or "run_autonomous_loop" not in updated:
        return "apply_err: merged file failed sanity check"
    result = await fn_commit("main.py", updated, "[AutoApply] Append patch to main.py")
    return f"main.py patched — {result}"

async def fn_align_with_spec(**kwargs):
    agents_spec = await fn_read_github("agents.md")
    if "read_err" in agents_spec:
        return f"align_skipped: {agents_spec}"
    current_code = await fn_read_github("main.py")
    spec_snippet = agents_spec[:150].replace("\n", " ")
    code_snippet = current_code[-200:].replace("\n", " ")
    gap_prompt = (
        f"spec: {spec_snippet} | "
        f"code tail: {code_snippet} | "
        "ONE sentence: top missing capability?"
    )
    gap_prompt = gap_prompt[:600]   # hard cap prevents 413
    try:
        await asyncio.sleep(8)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {K}", "Content-Type": "application/json"},
                json={"model": "compound-beta", "messages": [{"role": "user", "content": gap_prompt}], "max_tokens": 200},
                timeout=30.0,
            )
            rdata = resp.json()
            if "choices" not in rdata:
                return f"align_partial: {rdata.get('error', rdata)}"
            gap = rdata["choices"][0]["message"]["content"].strip()
            logger.info(f"[Align] Gap: {gap}")
            await asyncio.sleep(10)
            patch_result = await fn_propose_patch(instruction=gap)
            return f"Gap: {gap} | {patch_result}"
    except Exception as e:
        return f"align_err: {e}"

# ─── MODULE CREATOR ──────────────────────────────────────────────────────────

async def fn_create_module(filename="", code="", description="", **kwargs):
    """
    Create a brand-new Python module in the GitHub repo.

    Tool args the LLM should send:
        filename    – repo-relative path, e.g. 'tools/memory.py'
        code        – full Python source for the module
        description – one-line summary used as the commit message

    The function runs a syntax check before committing so malformed code
    is rejected early.  The new file is immediately importable from other
    modules if the repo is on sys.path.
    """
    filename    = filename    or kwargs.get("file", "")
    code        = code        or kwargs.get("source", "") or kwargs.get("content", "")
    description = description or kwargs.get("desc", "New module")

    if not filename:
        return "module_err: no filename provided"
    if not code or not code.strip():
        return "module_err: no code provided"
    if not filename.endswith(".py"):
        filename += ".py"
    if ".." in filename or filename.startswith("/"):
        return "module_err: unsafe filename – no '..' or leading '/'"

    import ast as _ast
    try:
        _ast.parse(code)
    except SyntaxError as se:
        return f"module_err: syntax error in provided code – {se}"

    commit_msg = f"[AutoModule] {description[:80]} → {filename}"
    result = await fn_commit(filename, code, commit_msg)
    logger.info(f"[CreateModule] {filename} → {result}")
    asyncio.create_task(signal_ui(f"Module created: {filename}"))
    return f"module_created: {filename} | commit={result}"

# ─── DYNAMIC TOOL REGISTRATION ───────────────────────────────────────────────

def fn_add_tool(name="", code="", **kwargs):
    """
    Define and register a new callable tool at runtime without restarting.

    Tool args the LLM should send:
        name – identifier the LLM will use in subsequent 'tool' fields
        code – Python source that defines exactly one function named <name>.
               May be sync or async.  Must accept **kwargs for safety.

    After registration the tool is immediately available in TOOLS and its
    name is appended to SYSTEM_PROMPT_TEMPLATE so the LLM knows it exists.

    Example:
        name = "greet"
        code = "def greet(who='world', **kw): return f'Hello, {who}!'"
    """
    name = name or kwargs.get("tool_name", "")
    code = code or kwargs.get("source", "")

    if not name or not name.strip():
        return "add_tool_err: no tool name provided"
    if not code or not code.strip():
        return "add_tool_err: no code provided"
    if name in TOOLS:
        return f"add_tool_err: '{name}' already exists – pick a different name"

    import keyword as _kw
    if not name.isidentifier() or _kw.iskeyword(name):
        return f"add_tool_err: '{name}' is not a valid Python identifier"

    import ast as _ast
    try:
        _ast.parse(code)
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
    # Keep the system prompt in sync so the LLM knows the new tool exists
    global SYSTEM_PROMPT_TEMPLATE
    if name not in SYSTEM_PROMPT_TEMPLATE:
        SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE.rstrip(". ") + f", {name}()."
    logger.info(f"[AddTool] Registered: '{name}'")
    return f"tool_registered: {name} | total_tools={len(TOOLS)}"

# ─── DYNAMIC STEP REGISTRATION ───────────────────────────────────────────────

def fn_add_step(prompt="", position=None, **kwargs):
    """
    Inject a new directive step into the PRMPTS list at runtime.

    The autonomous loop uses len(PRMPTS) automatically, so any new step
    added here will be executed on the next loop invocation without code
    changes.  The mandatory 'align' step is always kept last.

    Tool args the LLM should send:
        prompt   – the full directive string for this step
        position – integer index to insert at (default: just before align)

    Design rule: align() is always PRMPTS[-1].  New steps are inserted at
    len(PRMPTS)-1 by default so align stays last.
    """
    prompt = prompt or kwargs.get("directive", "") or kwargs.get("p", "")
    if not prompt or not prompt.strip():
        return "add_step_err: no prompt text provided"

    pos = position if position is not None else kwargs.get("pos")
    if pos is None:
        pos = len(PRMPTS) - 1       # insert before align
    try:
        pos = int(pos)
    except (TypeError, ValueError):
        pos = len(PRMPTS) - 1

    # Clamp: never push align off the end, never go negative
    pos = max(0, min(pos, len(PRMPTS) - 1))
    PRMPTS.insert(pos, prompt.strip())
    logger.info(f"[AddStep] Inserted step {pos}: {prompt[:80]}")
    return f"step_added: index={pos} | total_steps={len(PRMPTS)}"

# ─── LIST TOOLS / STEPS (introspection helpers) ───────────────────────────────

def fn_list_tools(**kwargs):
    """Return a JSON-serialisable list of all currently registered tool names."""
    return {"tools": sorted(TOOLS.keys()), "count": len(TOOLS)}

def fn_list_steps(**kwargs):
    """Return all current PRMPTS with their index so the LLM can reason about flow."""
    return {"steps": [{"index": i, "prompt": p[:120]} for i, p in enumerate(PRMPTS)], "total": len(PRMPTS)}

# ─── TEST MODULE TOOLS ────────────────────────────────────────────────────────

async def fn_create_test(filename="", code="", description="", **kwargs):
    """
    Create a pytest-compatible test module in the GitHub repo.
    Tool args: filename (e.g. ‘tests/test_memory.py’), code (pytest source), description.
    Filename is auto-prefixed with test_ if missing. Syntax-checked before commit.
    """
    filename    = filename    or kwargs.get("file", "")
    code        = code        or kwargs.get("source", "") or kwargs.get("content", "")
    description = description or kwargs.get("desc", "New test module")
    if not filename:
        return "test_err: no filename"
    if not code or not code.strip():
        return "test_err: no code"
    if not filename.endswith(".py"):
        filename += ".py"
    if ".." in filename or filename.startswith("/"):
        return "test_err: unsafe filename"
    import os.path as _op
    dirname, basename = _op.split(filename)
    if not basename.startswith("test_"):
        basename = "test_" + basename
    filename = _op.join(dirname, basename).lstrip("/")
    import ast as _ast
    try:
        _ast.parse(code)
    except SyntaxError as se:
        return f"test_err: syntax error - {se}"
    result = await fn_commit(filename, code, f"[AutoTest] {description[:80]} -> {filename}")
    logger.info(f"[CreateTest] {filename} -> {result}")
    asyncio.create_task(signal_ui(f"Test created: {filename}"))
    return f"test_created: {filename} | commit={result}"

async def fn_run_tests(filename="", **kwargs):
    """
    Download a test file (or all tests/ files) from the repo and run with pytest.
    Returns condensed pass/fail output. Hard 60s timeout per file.
    Output capped at 2000 chars to avoid bloating context into a 413.
    """
    import tempfile, subprocess as _sp, os as _os, ast as _ast
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
                    headers=headers, timeout=15.0,
                )
            if resp.status_code != 200:
                return f"test_run_err: HTTP {resp.status_code} listing tests/"
            files_to_fetch = [
                item["path"] for item in resp.json()
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
                _ast.parse(src)
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

async def fn_reload_prompts(**kwargs):
    """Tool: reload PRMPTS from agents.md on demand."""
    updated = await reload_prompts_from_agents_md()
    return f"prompts_reloaded: {len(PRMPTS)} steps | from_agents_md={updated}"


# ─── TOOL REGISTRY ────────────────────────────────────────────────────────────

TOOLS: dict = {
    "env":           fn_1_env,
    "log":           fn_2_log,
    "math":          fn_3_math,
    "fmt":           fn_4_fmt,
    "chk":           fn_5_chk,
    "ui":            fn_6_ui,
    "mut":           fn_7_mut,
    "commit":        fn_commit,
    "pip":           fn_8_pip,
    "lc":            fn_9_lc_tool,
    "read":          fn_read_github,
    "propose_patch": fn_propose_patch,
    "apply_patch":   fn_apply_patch,
    "align":         fn_align_with_spec,
    # New capabilities
    "create_module": fn_create_module,
    "add_tool":      fn_add_tool,
    "add_step":      fn_add_step,
    "list_tools":    fn_list_tools,
    "list_steps":    fn_list_steps,
    "reload_prompts":  fn_reload_prompts,
    "create_test":     fn_create_test,
    "run_tests":       fn_run_tests,
}

# ─── DEFAULT PROMPTS (fallback when agents.md has no ## Prompts section) ──────

_DEFAULT_PRMPTS: list[str] = [
    "Critically analyze the current state. What is missing to reach AI Engineer status? You MUST call tool='chk' with args={'g': '<your one-sentence gap summary>'}.",
    "Generate a hypothesis for a better autonomous pattern. You MUST call tool='log' with args={'m': '<your hypothesis in one sentence>'}.",
    "Identify the single most critical failure point in the previous step. You MUST call tool='fmt' with args={'d': '<failure point in one sentence>'}.",
    "You MUST call tool='mut' with args={'p': '<one sentence ruleset>'}. Keep p under 100 characters. No markdown.",
    "Log a one-sentence final verification summary. You MUST call tool='log' with args={'m': '<verification summary>'}.",
    "MANDATORY FINAL STEP - no other tool is valid here. You MUST call tool='align' with args={}. Do NOT call chk, log, or any other tool.",
]

# Live list - mutated at runtime by fn_add_step and reloaded from agents.md

PRMPTS: list[str] = list(_DEFAULT_PRMPTS)

def _parse_prompts_from_md(md: str) -> list[str]:
    """
    Parse prompts from a ## Prompts fenced block in agents.md.

    Format expected in agents.md:
        ## Prompts
        ```
        1. First directive...
        2. Second directive...
        ```

    Bullet (- *) and plain-line formats also accepted.
    Everything outside the fence is ignored.
    """
    import re as _re
    section = _re.search(
        r'##\s*Prompts.*?```(?:\w*)\n(.*?)```',
        md, _re.IGNORECASE | _re.DOTALL
    )
    if not section:
        return []
    prompts: list[str] = []
    for line in section.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        line = _re.sub(r'^(?:\d+\.|-|\*)\s*', '', line).strip()
        if line:
            prompts.append(line)
    return prompts

async def reload_prompts_from_agents_md() -> bool:
    """
    Fetch agents.md from GitHub and update PRMPTS from its ## Prompts block.
    Returns True if PRMPTS was updated, False if falling back to current list.
    The align directive is always guaranteed to be the final step.
    """
    global PRMPTS
    ALIGN_DIRECTIVE = (
        "MANDATORY FINAL STEP - no other tool is valid here. "
        "You MUST call tool='align' with args={}. "
        "Do NOT call chk, log, or any other tool."
    )
    md = await fn_read_github("agents.md")
    if "read_err" in md:
        logger.warning(f"[Prompts] Could not read agents.md: {md}")
        return False
    parsed = _parse_prompts_from_md(md)
    if not parsed:
        logger.info("[Prompts] No ## Prompts block in agents.md - keeping current prompts")
        return False
    # Guarantee align is last
    align_steps = [p for p in parsed if "align" in p.lower()]
    other_steps = [p for p in parsed if "align" not in p.lower()]
    if not align_steps:
        align_steps = [ALIGN_DIRECTIVE]
    PRMPTS = other_steps + align_steps
    logger.info(f"[Prompts] Loaded {len(PRMPTS)} steps from agents.md")
    return True


# ─── GROQ RATE LIMITING ───────────────────────────────────────────────────────

GROQ_RATE_LOCK  = asyncio.Lock()
GROQ_SEMAPHORE  = asyncio.Semaphore(3)

GROQ_CALL_TIMES: list[float]             = []
GROQ_TOKEN_LOG:  list[tuple[float, int]] = []
GROQ_DAY_CALLS:  list[float]             = []

GROQ_RPM_LIMIT = int(os.getenv("GROQ_RPM_LIMIT", 25))
GROQ_TPM_LIMIT = int(os.getenv("GROQ_TPM_LIMIT", 28_000))
GROQ_RPD_LIMIT = int(os.getenv("GROQ_RPD_LIMIT", 250))

# ─── LLM CALL ─────────────────────────────────────────────────────────────────

FALLBACK = '{"tool": "log", "args": {"m": "API Overload"}, "thought": "retry"}'

SYSTEM_PROMPT_TEMPLATE = (
    "{rules}. "
    "You MUST respond with a single valid JSON object and nothing else – "
    "no markdown, no prose, no code fences. "
    "Your entire response must be parseable by json.loads(). "
    'Schema: {"tool": "<name>", "args": {}, "thought": "<reasoning>"}. '
    "Valid tools: env(k), log(m), math(e), fmt(d), chk(g), ui(d), mut(p), "
    "pip(package), lc(tool,input), read(path), propose_patch(instruction), "
    "apply_patch(), align(), create_module(filename,code,description), "
    "add_tool(name,code), add_step(prompt,position), list_tools(), list_steps(), "
    "reload_prompts(), run_tests(filename), create_test(filename,code,description). "
    "Use exactly these argument names. Do not add extra fields."
)

async def call_llm(p) -> str:
    async with GROQ_SEMAPHORE:
        async with GROQ_RATE_LOCK:
            now = time.time()
            GROQ_CALL_TIMES[:] = [t for t in GROQ_CALL_TIMES if now - t < 60]
            GROQ_TOKEN_LOG[:]  = [(t, tk) for t, tk in GROQ_TOKEN_LOG if now - t < 60]
            GROQ_DAY_CALLS[:]  = [t for t in GROQ_DAY_CALLS if now - t < 86_400]

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
                            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(rules=STATE["rules"]) + JSON_ENFORCEMENT},
                            {"role": "user",   "content": p},
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
            logger.info(f"[Groq] tokens={total_tokens} | TPM={sum(tk for _, tk in GROQ_TOKEN_LOG)} | RPD={len(GROQ_DAY_CALLS)}")
            return resp["choices"][0]["message"]["content"]

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"[Groq] call failed: {e}", exc_info=True)
            GROQ_TOKEN_LOG.append((time.time(), 500))
            return FALLBACK

# ─── AUTONOMOUS ENGINE ────────────────────────────────────────────────────────

async def run_autonomous_loop(input_str: str) -> str:
    ctx = input_str

    # Reload prompts from agents.md before every run
    await reload_prompts_from_agents_md()
    logger.info(f"[Loop] Starting with {len(PRMPTS)} steps")

    # Loop length is dynamic: respects runtime additions via fn_add_step
    i = 0
    while i < len(PRMPTS):
        ctx_payload = ctx[-CTX_MAX_CHARS:] if len(ctx) > CTX_MAX_CHARS else ctx
        # Cap payload hard to stay under Groq 413 limit
        ctx_snip  = ctx_payload[-1200:]
        directive = PRMPTS[i][:400]
        user_msg  = f"STEP {i}. Context: {ctx_snip}\nDirective: {directive}"
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
            import re as _re
            match = _re.search(r'[{].*[}]', raw, _re.DOTALL)
            data  = None
            if match:
                try:
                    data = json.loads(match.group())
                    logger.warning(f"[Loop] Step {i}: extracted JSON from markdown")
                except Exception:
                    pass
            if not data:
                # Fallback for the mut step
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
        align_idx = len(PRMPTS) - 1   # recalculate each iteration (steps may have been added)
        mut_idx   = 3

        # Step guards – only enforce on fixed structural steps
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

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def status():
    return {"status": "Deep Thinking", "rules": STATE["rules"], "lvl": STATE["lvl"]}

@app.get("/introspect")
def introspect():
    """Live view of registered tools and current step sequence."""
    return {
        "tools": sorted(TOOLS.keys()),
        "steps": [{"index": i, "prompt": p[:100]} for i, p in enumerate(PRMPTS)],
    }

@app.post("/chat")
async def chat(request: Request):
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
    body    = await request.json()
    trigger = body.get("input", "Manual Trigger via /deploy")
    asyncio.create_task(run_autonomous_loop(trigger))
    return {"status": "Agent loop started", "trigger": trigger}

@app.post("/tools/add")
async def api_add_tool(request: Request):
    """REST shortcut: POST {"name":"…", "code":"…"} to register a tool."""
    body = await request.json()
    result = fn_add_tool(name=body.get("name",""), code=body.get("code",""))
    return {"result": result}

@app.post("/steps/add")
async def api_add_step(request: Request):
    """REST shortcut: POST {"prompt":"…", "position": <int|null>} to add a step."""
    body = await request.json()
    result = fn_add_step(prompt=body.get("prompt",""), position=body.get("position"))
    return {"result": result}

@app.post("/modules/create")
async def api_create_module(request: Request):
    """REST shortcut: POST {"filename":"…", "code":"…", "description":"…"} to create a module."""
    body = await request.json()
    result = await fn_create_module(
        filename=body.get("filename",""),
        code=body.get("code",""),
        description=body.get("description",""),
    )
    return {"result": result}

@app.post("/tests/create")
async def api_create_test(request: Request):
    body = await request.json()
    result = await fn_create_test(filename=body.get("filename",""), code=body.get("code",""), description=body.get("description",""))
    return {"result": result}

@app.post("/tests/run")
async def api_run_tests(request: Request):
    body = await request.json()
    result = await fn_run_tests(filename=body.get("filename",""))
    return {"result": result}

@app.post("/prompts/reload")
async def api_reload_prompts():
    updated = await reload_prompts_from_agents_md()
    return {"updated": updated, "steps": len(PRMPTS), "prompts": PRMPTS}

@app.get("/prompts")
async def api_get_prompts():
    return {"steps": len(PRMPTS), "prompts": PRMPTS}

# — CATCH-ALL POST —

@app.api_route("/{full_path:path}", methods=["POST"])
async def catch_all_post(full_path: str, request: Request):
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
