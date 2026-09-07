// Captures the dashboard screenshots used in the README.
// Playwright is not a project dependency; install it only if you need to regenerate them:
//   cd frontend && npm i --no-save playwright && npx playwright install chromium
//   cp ../scripts/capture_screenshots.mjs ./_shot.mjs   (must sit beside node_modules)
//   CHROME_EXE=<path to chromium> node ./_shot.mjs ../docs/screenshots && rm _shot.mjs
// Both servers must be running (make serve, and npm run dev in frontend).

import { chromium } from "playwright";

const out = process.argv[2];
const exe = process.env.CHROME_EXE;
const browser = await chromium.launch({ executablePath: exe });
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 }, deviceScaleFactor: 2 });
const errors = [];
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
page.on("pageerror", (e) => errors.push(String(e)));

await page.goto("http://127.0.0.1:3000/", { waitUntil: "networkidle", timeout: 120000 });
await page.waitForTimeout(6000);

const text = (await page.innerText("body")).toLowerCase();
console.log("SECTIONS:", ["Overview","Forecast","Price and signals","Market regime","Model performance","Why this prediction","Historical predictions","Error analysis","Limitations"].filter(s => text.includes(s.toLowerCase())).join(" | "));
console.log("HAS_PRICE:", /\$\d/.test(text));
console.log("HAS_PROBS:", /\d\d\.\d%/.test(text));
console.log("CANVAS:", await page.locator("canvas").count());
console.log("ERRORS:", errors.length ? errors.slice(0,5).join(" // ") : "none");
console.log("DROPDOWN_OPTIONS:", await page.locator("select option").count());
console.log("TABLE_ROWS:", await page.locator("tbody tr").count());

await page.screenshot({ path: `${out}/dashboard-full.png`, fullPage: true });
for (const [id, name] of [["overview","overview"],["forecast","forecast"],["price","price"],["regime","regime"],["performance","performance"],["explain","explain"],["predictions","predictions"],["errors","errors"]]) {
  const el = page.locator(`#${id}`);
  if (await el.count()) await el.screenshot({ path: `${out}/${name}.png` });
}
await browser.close();
