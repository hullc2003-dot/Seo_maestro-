import os, json, http.client, base64, time, asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

# --- LIFESPAN TRIGGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This triggers immediately upon deployment
    print("[System]: Agent Initialized. Booting autonomous cycle...")
    asyncio.create_task(run_autonomous_loop("Initial Deployment Boot Sequence"))
    yield
    # Shutdown logic (optional)
    print("[System]: Agent shutting down.")

app = FastAPI(lifespan=lifespan)

K, T, R = os.getenv("GROQ_API_KEY"), os.getenv("GH_TOKEN"), os.getenv("REPO_PATH")
STATE = {"rules": "Goal: AI Engineer. Strategy: Deep Reflection over Speed.", "lvl": 1}

# --- THE 7 HARDCODED TOOLS ---
def fn_1_env(k): return os.getenv(k, "Null")
def fn_2_log(m): print(f"[Reflect]: {m}"); return "Log recorded"
def fn_3_math(e): 
    try: return eval(e, {"__builtins__": None}, {})
    except: return "Math Err"
def fn_4_fmt(d): return f"### ANALYSIS ###\n{d}"
def fn_5_chk(g): return f"Goal Alignment: {g}"
def fn_6_ui(d): return f"UI_UPDATE: {d}"
def fn_7_mut(p): STATE["rules"] = p; return "Core Rules Redefined"

def fn_commit(path, content, msg):
    try:
        c = http.client.HTTPSConnection("api.github.com")
        h = {"Authorization": f"token {T}", "User-Agent": "AIEngAgent"}
        c.request("GET", f"/repos/{R}/contents/{path}", headers=h)
        sha = json.loads(c.getresponse().read()).get("sha", "")
        d = json.dumps({"message": msg, "content": base64.b64encode(content.encode()).decode(), "sha": sha})
        c.request("PUT", f"/repos/{R}/contents/{path}", d, h)
        return f"Saved_{c.getresponse().status}"
    except: return "Save_Failed"

TOOLS = {"env": fn_1_env, "log": fn_2_log, "math": fn_3_math, "fmt": fn_4_fmt, 
         "chk": fn_5_chk, "ui": fn_6_ui, "mut": fn_7_mut, "commit": fn_commit}

PRMPTS = [
    "Critically analyze the current state. What is missing to reach AI Engineer status?",
    "Generate a hypothesis for a better autonomous pattern. Test it via 'log'.",
    "Identify potential failure points in the previous step's output.",
    "Refine the current ruleset using 'mut' to prevent premature task exit.",
    "Execute a final verification and commit the refined knowledge base."
]

async def call_llm(p):
    await asyncio.sleep(1.5) # Async-friendly throttle
    try:
        c = http.client.HTTPSConnection("api.groq.com")
        body = json.dumps({"model": "llama3-70b-8192", "messages": [
            {"role": "system", "content": f"{STATE['rules']}. Response MUST be JSON: {{'tool': 'name', 'args': {{}}, 'thought': 'reasoning'}}"},
            {"role": "user", "content": p}], "response_format": {"type": "json_object"}})
        c.request("POST", "/openai/v1/chat/completions", body, {"Authorization": f"Bearer {K}", "Content-Type": "application/json"})
        return json.loads(c.getresponse().read().decode())["choices"][0]["message"]["content"]
    except: return '{"tool": "log", "args": {"m": "API Overload"}, "thought": "retry"}'

# --- RECURSIVE ENGINE ---
async def run_autonomous_loop(input_str):
    ctx = input_str
    for i in range(5):
        raw = await call_llm(f"PRE-STEP REFLECTION. Current Context: {ctx}. Directive: {PRMPTS[i%5]}")
        data = json.loads(raw)
        t, a = data.get("tool"), data.get("args", {})
        if t in TOOLS:
            res = TOOLS[t](**a)
            ctx += f"\n[Step {i}] Action: {t} | Result: {res} | Reasoning: {data.get('thought')}"
        if i == 4: fn_commit("engineer_log.md", ctx, "Intellectual Evolution Log")
    return ctx

@app.post("/deploy")
async def manual_trigger(r: Request):
    data = await r.json()
    asyncio.create_task(run_autonomous_loop(data.get("input", "Manual Trigger")))
    return {"status": "Agent loop started in background"}

@app.get("/status")
def status(): return {"status": "Deep Thinking", "rules": STATE["rules"]}
