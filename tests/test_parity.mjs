// Proves the browser engine gives the same probabilities as the Python model.
// Run:  node tests/test_parity.mjs     (after: python improved/export_web.py ...)
import fs from "node:fs";
import "../docs/engine.js";

const model = JSON.parse(fs.readFileSync(new URL("../docs/model.json", import.meta.url)));
const cases = JSON.parse(fs.readFileSync(new URL("../docs/parity.json", import.meta.url)));

let worst = 0;
for (const c of cases) {
  const r = globalThis.FND.analyse(model, c.title, c.text);
  const diff = Math.abs(r.pReal - c.p_real);
  worst = Math.max(worst, diff);
  if (diff > 1e-3) {
    console.error(`MISMATCH js=${r.pReal.toFixed(5)} py=${c.p_real.toFixed(5)} :: ${(c.title + " " + c.text).slice(0, 70)}`);
    process.exitCode = 1;
  }
}
console.log(`${cases.length} cases, worst |js - python| = ${worst.toExponential(2)}`);
if (process.exitCode) console.error("PARITY FAILED"); else console.log("PARITY OK");
