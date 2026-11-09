// Robustness checks for the browser engine: it must never throw and must stay in [0, 1].
import fs from "node:fs";
import "../docs/engine.js";

const model = JSON.parse(fs.readFileSync(new URL("../docs/model.json", import.meta.url)));
const big = "The Senate voted on budget policy. ".repeat(20000);
const cases = {
  empty: "", spaces: "   \n\t ", punctuation: "!!!???...", digits: "1234 5678",
  oneWord: "hi", emoji: "\u{1F600}\u{1F4A5} breaking news \u{1F680}", arabic: "العراق يمنع",
  html: "<script>alert(1)</script><p>The Senate voted</p>", url: "http://a.b/c www.x.org/y",
  proto: "__proto__ constructor toString hasOwnProperty", huge: big, nul: "abc\u0000def ghi",
  allCaps: "BREAKING SHOCKING BOMBSHELL EXPOSED", reutersOnly: "WASHINGTON (Reuters) -",
};
let bad = 0, n = 0;
for (const [name, t] of Object.entries(cases)) {
  n++;
  try {
    const r = globalThis.FND.analyse(model, "", t);
    if (!(r.pReal >= 0 && r.pReal <= 1) || Number.isNaN(r.pReal)) throw new Error("pReal out of range: " + r.pReal);
  } catch (e) { bad++; console.error("FAIL", name, e.message); }
}
// determinism and monotonic sanity: shouting conspiracy text should score lower than a wire-style report
const fakeP = globalThis.FND.analyse(model, "", "SHOCKING! The elites hide the truth, share before the mainstream media deletes this!").pReal;
const realP = globalThis.FND.analyse(model, "", "The Senate voted 52-48 on Tuesday to approve the budget bill, officials said.").pReal;
if (!(fakeP < realP)) { bad++; console.error("FAIL ordering", fakeP, realP); }
console.log(`${n} robustness cases + ordering check, failures: ${bad}`);
process.exit(bad ? 1 : 0);
