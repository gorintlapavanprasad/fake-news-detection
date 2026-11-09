"""
Web front end for the fake-news checker.

    python app.py            ->  http://127.0.0.1:5000

Endpoints:
    GET  /            the UI
    POST /api/check   JSON {"title": "...", "text": "..."} -> verdict + explanation
    GET  /api/health  model info
"""

import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

import fnd

MODEL_PATH = os.environ.get("FND_MODEL", os.path.join(os.path.dirname(__file__), "model.joblib"))
bundle = joblib.load(MODEL_PATH)
pipe = bundle["model"]
tfidf, clf = pipe.steps[0][1], pipe.steps[-1][1]
vocab = np.array(tfidf.get_feature_names_out())

app = Flask(__name__)
MIN_WORDS = 25  # below this, predictions are unreliable


def analyse(title: str, text: str) -> dict:
    df = pd.DataFrame({"title": [title], "text": [text]})
    cleaned = fnd.build_field(df, bundle["field"], bundle["max_words"], True)
    words = len(cleaned.iloc[0].split())

    if hasattr(clf, "predict_proba"):
        p_real = float(pipe.predict_proba(cleaned)[0][1])
    else:
        d = float(pipe.decision_function(cleaned)[0])
        p_real = float(1 / (1 + np.exp(-2 * d)))

    # Explanation: which words pushed the score toward REAL / FAKE (linear models only).
    reasons = {"real": [], "fake": []}
    if hasattr(clf, "coef_"):
        row = tfidf.transform(cleaned)
        contrib = row.multiply(clf.coef_[0]).tocoo()
        pairs = sorted(zip(contrib.col, contrib.data), key=lambda x: x[1])
        reasons["fake"] = [(vocab[c], round(float(v), 3)) for c, v in pairs[:8] if v < 0]
        reasons["real"] = [(vocab[c], round(float(v), 3)) for c, v in pairs[::-1][:8] if v > 0]

    label = "REAL" if p_real >= 0.5 else "FAKE"
    confidence = max(p_real, 1 - p_real)
    warnings = []
    if words < MIN_WORDS:
        warnings.append(f"Only {words} usable words - results for very short text are unreliable.")
    if confidence < 0.7:
        warnings.append("The model is not confident. Treat this as 'unclear' and check other sources.")
    warnings.append("This tool detects the WRITING STYLE of known fake/real news. "
                    "It cannot verify facts. Always cross-check important claims.")
    return {"label": label, "p_real": round(p_real, 4), "confidence": round(confidence, 4),
            "unclear": confidence < 0.7 or words < 10, "words": words, "reasons": reasons,
            "warnings": warnings}


@app.get("/api/health")
def health():
    return jsonify(status="ok", model=bundle["id"], field=bundle["field"])


@app.post("/api/check")
def check():
    data = request.get_json(silent=True) or {}
    title, text = str(data.get("title", "")), str(data.get("text", ""))
    if not (title + text).strip():
        return jsonify(error="Please enter a headline or article text."), 400
    if len(title) + len(text) > 100_000:
        return jsonify(error="Text too long (max 100,000 characters)."), 413
    return jsonify(analyse(title, text))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fake News Checker</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--ink:#e2e8f0;--mute:#94a3b8;--real:#22c55e;--fake:#ef4444;--warn:#f59e0b;--acc:#6366f1}
*{box-sizing:border-box}body{margin:0;font:16px/1.5 system-ui,Segoe UI,sans-serif;background:var(--bg);color:var(--ink)}
main{max-width:860px;margin:0 auto;padding:32px 20px}
h1{margin:0 0 4px;font-size:2rem}p.sub{margin:0 0 24px;color:var(--mute)}
.card{background:var(--card);border-radius:14px;padding:20px;margin-bottom:20px}
input,textarea{width:100%;background:#0b1220;color:var(--ink);border:1px solid #334155;border-radius:10px;padding:12px;font:inherit}
textarea{min-height:200px;resize:vertical}label{display:block;margin:12px 0 6px;color:var(--mute);font-size:.9rem}
.row{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}
button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:12px 22px;font-weight:600;font-size:1rem;font-family:inherit;cursor:pointer}
button.ghost{background:#334155}button:disabled{opacity:.5;cursor:wait}
#result{display:none}.verdict{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.badge{font-size:1.8rem;font-weight:800;padding:8px 22px;border-radius:12px;color:#fff}
.REAL{background:var(--real)}.FAKE{background:var(--fake)}.UNCLEAR{background:var(--warn)}
.bar{height:14px;border-radius:7px;background:linear-gradient(90deg,var(--fake),#334155 50%,var(--real));position:relative;margin:18px 0 4px}
.bar i{position:absolute;top:-5px;width:4px;height:24px;background:#fff;border-radius:2px;transform:translateX(-2px)}
.scale{display:flex;justify-content:space-between;color:var(--mute);font-size:.8rem}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:600px){.cols{grid-template-columns:1fr}}
.chip{display:inline-block;margin:3px;padding:3px 10px;border-radius:99px;font-size:.85rem}
.chip.r{background:#14532d}.chip.f{background:#7f1d1d}
.warn{color:var(--warn);font-size:.9rem;margin:6px 0}.err{color:var(--fake)}
h3{margin:0 0 8px;font-size:1rem;color:var(--mute);font-weight:600}
</style></head><body><main>
<h1>Fake News Checker</h1>
<p class="sub">Paste a headline or article. The model scores how closely its writing matches known real vs fake news.</p>
<div class="card">
 <label for="title">Headline (optional)</label><input id="title" placeholder="e.g. Senate passes budget bill">
 <label for="text">Article text</label><textarea id="text" placeholder="Paste the article here..."></textarea>
 <div class="row"><button id="go">Check</button>
  <button class="ghost" data-ex="real">Try a real-style example</button>
  <button class="ghost" data-ex="fake">Try a fake-style example</button>
  <button class="ghost" id="clear">Clear</button></div>
 <p id="err" class="err"></p>
</div>
<div class="card" id="result">
 <div class="verdict"><span class="badge" id="badge"></span><div><div id="conf" style="font-size:1.2rem;font-weight:700"></div><div id="meta" style="color:var(--mute)"></div></div></div>
 <div class="bar"><i id="mark"></i></div><div class="scale"><span>FAKE</span><span>REAL</span></div>
 <div class="cols" style="margin-top:18px"><div><h3>Words pushing toward REAL</h3><div id="rw"></div></div><div><h3>Words pushing toward FAKE</h3><div id="fw"></div></div></div>
 <div id="warns" style="margin-top:14px"></div>
</div>
</main>
<script>
const $=id=>document.getElementById(id);
const EX={real:["Senate passes budget bill","The Senate voted 52-48 on Tuesday to approve a federal budget bill after weeks of negotiation between party leaders, sending it to the House for consideration, officials said."],
fake:["SHOCKING: Secret elites control the weather","You will not believe what they are hiding! Insiders reveal the shadowy elite use machines to control storms. Share this before it gets deleted by the mainstream media!!!"]};
document.querySelectorAll("[data-ex]").forEach(b=>b.onclick=()=>{const e=EX[b.dataset.ex];$("title").value=e[0];$("text").value=e[1]});
$("clear").onclick=()=>{$("title").value=$("text").value="";$("result").style.display="none";$("err").textContent=""};
function chips(el,list,cls){el.replaceChildren(...(list.length?list:[["(none)",0]]).map(([w])=>{const s=document.createElement("span");s.className="chip "+cls;s.textContent=w;return s}))}
$("go").onclick=async()=>{
 $("err").textContent="";$("go").disabled=true;
 try{const r=await fetch("/api/check",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({title:$("title").value,text:$("text").value})});
  const d=await r.json();if(!r.ok)throw new Error(d.error||"Request failed");
  const lab=d.unclear?"UNCLEAR":d.label;$("badge").textContent=lab;$("badge").className="badge "+lab;
  $("conf").textContent=(d.unclear?"Low confidence: ":"")+Math.round(d.confidence*100)+"% "+d.label;
  $("meta").textContent=d.words+" words analysed";$("mark").style.left=(d.p_real*100)+"%";
  chips($("rw"),d.reasons.real,"r");chips($("fw"),d.reasons.fake,"f");
  $("warns").replaceChildren(...d.warnings.map(w=>{const p=document.createElement("div");p.className="warn";p.textContent="⚠ "+w;return p}));
  $("result").style.display="block";$("result").scrollIntoView({behavior:"smooth"});
 }catch(e){$("err").textContent=e.message}finally{$("go").disabled=false}};
</script></body></html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)))
