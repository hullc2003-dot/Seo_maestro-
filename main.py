import os, json, http.client, base64, time
from fastapi import FastAPI, Request

app = FastAPI()
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

# The List: fn_1_env, fn_2_log, fn_3_math, fn_4_fmt, fn_5_chk, fn_6_ui, fn_7_mut.

# Deep-Reasoning Prompt Embedded Set
PRMPTS = [
    "Critically analyze the current state. What is missing to reach AI Engineer status?",
    "Generate a hypothesis for a better autonomous pattern. Test it via 'log'.",
    "Identify potential failure points in the previous step's output.",
    "Refine the current ruleset using 'mut' to prevent premature task exit.",
    "Execute a final verification and commit the refined knowledge base."
]

async def call_llm(p):
    time.sleep(1.5) # Intellectual throttle for Groq RPM limits
    try:
        c = http.client.HTTPSConnection("api.groq.com")
        body = json.dumps({"model": "llama3-70b-8192", "messages": [ # Using 70B for better reasoning
            {"role": "system", "content": f"{STATE['rules']}. Focus on accuracy. Response MUST be valid JSON: {{'tool': 'name', 'args': {{}}, 'thought': 'reasoning'}}"},
            {"role": "user", "content": p}], "response_format": {"type": "json_object"}})
        c.request("POST", "/openai/v1/chat/completions", body, {"Authorization": f"Bearer {K}", "Content-Type": "application/json"})
        return json.loads(c.getresponse().read().decode())["choices"][0]["message"]["content"]
    except: return '{"tool": "log", "args": {"m": "API Overload - Cooling down"}, "thought": "retrying"}'

@app.post("/deploy")
async def execute(r: Request):
    ctx = (await r.json()).get("input", "Begin Engineer sequence.")
    for i in range(5):
        # The 'Intellectual Race' Step: Ask the model to reflect on the Context BEFORE acting
        raw = await call_llm(f"PRE-STEP REFLECTION. Current Context: {ctx}. Directive: {PRMPTS[i%5]}")
        data = json.loads(raw)
        t, a = data.get("tool"), data.get("args", {})
        
        if t in TOOLS:
            res = TOOLS[t](**a)
            # Accumulate context instead of replacing it (Deep Memory)
            ctx += f"\n[Step {i}] Action: {t} | Result: {res} | Reasoning: {data.get('thought')}"
        
        if i == 4: TOOLS["commit"]("engineer_log.md", ctx, "Intellectual Evolution Log")
    return {"status": "Analysis Complete", "final_knowledge": ctx}

@app.get("/status")
def status(): return {"status": "Deep Thinking", "rules": STATE["rules"]}
