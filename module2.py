# module2.py: Tools and Functions
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

from module1 import K, T, R, STATE, CTX_MAX_CHARS, signal_ui, logger

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
    try:
        score = globals().get("alignment_score", 1.0)
        return float(score) >= threshold
    except Exception:
        logger.exception("[fn_5_chk] Alignment check failed")
        return False

def fn_6_ui(d="", **kwargs): return f"UI_UPDATE: {d or json.dumps(kwargs)}"

def fn_7_mut(new_rule: str) -> None:
    global operational_rules
    if not isinstance(operational_rules, list):
        operational_rules = []
    operational_rules.append(new_rule)
    logger.info("[fn_7_mut] Added new rule: %s", new_rule)
    asyncio.create_task(signal_ui("Operational rules mutated"))

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

def fn_auto_gen_code

    def _render_template(req: Dict) -> str:
        """Create a simple function based on a high‑level requirement."""
        name = req.get("func", "generated_func")
        params = req.get("params", [])
        body = req.get("body", "return None")
        param_str = ", ".join(params)
        return f"def {name}({param_str}):\n    {body}\n"

    def generate_code(requirements: Dict) -> str:
        """Generate Python source code from a requirement dict."""
    # Very basic templating – can be expanded with LLM calls later.
        return _render_template(requirements)

    def _load_function(code: str, func_name: str) -> Callable:
        """Dynamically load the generated function."""
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        return namespace[func_name]

    def test_code(code: str, requirements: Dict) -> bool:
        """Run supplied test cases against the generated function."""
        tests = requirements.get("tests", [])
        if not tests:
        return True  # No tests to run
        func_name = requirements.get("func", "generated_func")
        try:
            func = _load_function(code, func_name)
        except Exception:
            return False
        for case in tests:
            args = case.get("args", [])
            expected = case.get("expected")
        try:
            result = func(*args)
        except Exception:
            return False
        if result != expected:
            return False
            return True

    def validate_code(code: str, requirements: Dict) -> bool:
          """Validate that the generated code meets structural requirements."""
          # Example: ensure required function name exists
          required_name = requirements.get("func")
          if required_name and f"def {required_name}(" not in code:
          return False
          return True

    def autonomous_code_dev(requirements: Dict) -> str:
        """Generate, test, and validate code autonomously."""
         code = generate_code(requirements)
        if not test_code(code, requirements):
            return "Testing failed"
        if not validate_code(code, requirements):
            return "Validation failed"
        return code

    def main():
    # Example high‑level requirement
        req = {
            "func": "add",
            "params": ["a", "b"],
            "body": "return a + b",
            "tests": [
                {"args": [1, 2], "expected": 3},
                {"args": [-1, 5], "expected": 4}
        ]
    }
        result = await autonomous_code_dev(req)
        print(result)

    asyncio.run(main())


async def fn_create_module(filename="", code="", description="", **kwargs):
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

    try:
        ast.parse(code)
    except SyntaxError as se:
        return f"module_err: syntax error in provided code – {se}"

    commit_msg = f"[AutoModule] {description[:80]} → {filename}"
    result = await fn_commit(filename, code, commit_msg)
    logger.info(f"[CreateModule] {filename} → {result}")
    asyncio.create_task(signal_ui(f"Module created: {filename}"))
    return f"module_created: {filename} | commit={result}"

def fn_add_tool(name="", code="", **kwargs):
    from module1 import SYSTEM_PROMPT_TEMPLATE
    global SYSTEM_PROMPT_TEMPLATE
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
    if name not in SYSTEM_PROMPT_TEMPLATE:
        SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_TEMPLATE.rstrip(". ") + f", {name}()."
    logger.info(f"[AddTool] Registered: '{name}'")
    return f"tool_registered: {name} | total_tools={len(TOOLS)}"

def fn_add_step(prompt="", position=None, **kwargs):
    from module3 import PRMPTS
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

    pos = max(0, min(pos, len(PRMPTS) - 1))
    PRMPTS.insert(pos, prompt.strip())
    logger.info(f"[AddStep] Inserted step {pos}: {prompt[:80]}")
    return f"step_added: index={pos} | total_steps={len(PRMPTS)}"

def fn_list_tools(**kwargs):
    return {"tools": sorted(TOOLS.keys()), "count": len(TOOLS)}

def fn_list_steps(**kwargs):
    from module3 import PRMPTS
    return {"steps": [{"index": i, "prompt": p[:120]} for i, p in enumerate(PRMPTS)], "total": len(PRMPTS)}

async def fn_create_test(filename="", code="", description="", **kwargs):
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
    dirname, basename = os.path.split(filename)
    if not basename.startswith("test_"):
        basename = "test_" + basename
    filename = os.path.join(dirname, basename).lstrip("/")
    try:
        ast.parse(code)
    except SyntaxError as se:
        return f"test_err: syntax error - {se}"
    result = await fn_commit(filename, code, f"[AutoTest] {description[:80]} -> {filename}")
    logger.info(f"[CreateTest] {filename} -> {result}")
    asyncio.create_task(signal_ui(f"Test created: {filename}"))
    return f"test_created: {filename} | commit={result}"

async def fn_run_tests(filename="", **kwargs):
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
            basename = os.path.basename(fpath)
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
            local_path = os.path.join(tmpdir, basename)
            with open(local_path, "w", encoding="utf-8") as fh:
                fh.write(src)
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "pytest", local_path,
                     "-v", "--tb=short", "--no-header", "-q"],
                    capture_output=True, text=True, timeout=60, cwd=tmpdir,
                )
                out_lines = (proc.stdout + proc.stderr).strip().splitlines()
                condensed = "\n".join(out_lines[-40:])
                status = "PASS" if proc.returncode == 0 else "FAIL"
                results.append(f"{status} [{fpath}]\n{condensed}")
            except subprocess.TimeoutExpired:
                results.append(f"TIMEOUT [{fpath}]: exceeded 60s")
            except Exception as exc:
                results.append(f"RUN_ERR [{fpath}]: {exc}")

    summary = "\n\n".join(results)
    logger.info(f"[RunTests] {len(files_to_fetch)} file(s) run")
    asyncio.create_task(signal_ui(f"Tests done: {len(results)} file(s)"))
    return summary[:2000]

async def reload_prompts_from_agents_md() -> bool:
    from module3 import PRMPTS
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
    align_steps = [p for p in parsed if "align" in p.lower()]
    other_steps = [p for p in parsed if "align" not in p.lower()]
    if not align_steps:
        align_steps = [ALIGN_DIRECTIVE]
    PRMPTS = other_steps + align_steps
    logger.info(f"[Prompts] Loaded {len(PRMPTS)} steps from agents.md")
    return True

def _parse_prompts_from_md(md: str) -> list[str]:
    section = re.search(
        r'##\s*Prompts.*?```(?:\w*)\n(.*?)```',
        md, re.IGNORECASE | re.DOTALL
    )
    if not section:
        return []
    prompts: list[str] = []
    for line in section.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^(?:\d+\.|-|\*)\s*', '', line).strip()
        if line:
            prompts.append(line)
    return prompts

async def fn_reload_prompts(**kwargs):
    updated = await reload_prompts_from_agents_md()
    from module3 import PRMPTS
    return f"prompts_reloaded: {len(PRMPTS)} steps | from_agents_md={updated}"

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
    "auto_code_gen": fn_auto_code_gen   
    "create_module": fn_create_module,
    "add_tool":      fn_add_tool,
    "add_step":      fn_add_step,
    "list_tools":    fn_list_tools,
    "list_steps":    fn_list_steps,
    "reload_prompts":  fn_reload_prompts,
    "create_test":     fn_create_test,
    "run_tests":       fn_run_tests,
}
