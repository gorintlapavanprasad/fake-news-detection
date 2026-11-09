// Honest accuracy check of the browser engine on real data.
// Uses a model trained on the train split only (build/model_split.json) and held-out
// articles from the original and the new dataset, plus an unseen third source.
// Generate the files with:  python improved/export_web.py ...
// Run:  node tests/eval_js_engine.mjs
import fs from "node:fs";
import "../docs/engine.js";

const read = (p) => JSON.parse(fs.readFileSync(new URL(p, import.meta.url)));
const model = read("../build/model_split.json");
const sets = read("../build/eval.json");
const py = read("../build/eval_python.json");

let failed = false;
console.log("set".padEnd(20), "n".padStart(5), "js acc".padStart(8), "python".padStart(8), "  fake recall  real recall");
for (const [name, rows] of Object.entries(sets)) {
  let ok = 0, tp = 0, tn = 0, nf = 0, nr = 0;
  for (const r of rows) {
    const pred = globalThis.FND.analyse(model, r.title, r.text).pReal >= 0.5 ? 1 : 0;
    if (pred === r.label) ok++;
    if (r.label === 0) { nf++; if (pred === 0) tn++; } else { nr++; if (pred === 1) tp++; }
  }
  const acc = ok / rows.length;
  console.log(name.padEnd(20), String(rows.length).padStart(5), acc.toFixed(4).padStart(8),
    String(py[name]).padStart(8), "  ", (tn / nf).toFixed(3).padStart(8), (tp / nr).toFixed(3).padStart(12));
  // the JS engine must reproduce the Python accuracy (same model, same texts)
  if (Math.abs(acc - py[name]) > 0.003) { console.error("  JS/Python accuracy gap on", name); failed = true; }
}
process.exit(failed ? 1 : 0);
