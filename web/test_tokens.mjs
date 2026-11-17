// Checks that the demo's tokens (dizione-ptbr in TypeScript + tokens.js) equal tiny_voca.text.phonemize in Python.
//   node web/test_tokens.mjs tokens_golden.json
import { readFileSync } from "node:fs";
import { Dizione } from "./public/assets/dizione/engine.js";
import * as symbols from "./public/assets/dizione/symbols.js";
import { makeTokenizer } from "./public/tokens.js";
const data = new URL("./public/assets/dizione/data/", import.meta.url);
const d = new Dizione(readFileSync(new URL("lexicon.tsv", data), "utf8"), JSON.parse(readFileSync(new URL("open_table.json", data), "utf8")));
const tokenize = makeTokenizer(symbols);
const gold = JSON.parse(readFileSync(process.argv[2], "utf8"));
let bad = 0;
for (const [text, ids, stress] of gold) {
  const r = tokenize(d.pieces(text));
  const same = ids === null ? r === null : r !== null && JSON.stringify(r.ids) === JSON.stringify(ids) && JSON.stringify(r.stress) === JSON.stringify(stress);
  if (!same && bad++ < 5) console.log("DIFF", JSON.stringify(text), "\n py", ids && ids.join(","), "\n js", r && r.ids.join(","));
}
console.log(`${gold.length - bad}/${gold.length} identical`);
process.exit(bad ? 1 : 0);
