// Offline check of the trigger: fetch is stubbed, so no network and no real token.
// Run from the repo root: node trigger/test.mjs
import assert from "node:assert/strict";
import worker, { WORKFLOW_URL } from "./index.js";

const tick = { cron: "7,27,47 * * * *", scheduledTime: Date.now() };
const calls = [];
globalThis.fetch = async (url, init) => {
  calls.push({ url, init });
  return new Response(null, { status: 204 });
};

// One dispatch of the tournament workflow on main, marked as the Cloudflare clock
await worker.scheduled(tick, { GITHUB_TOKEN: "fake-token" });
assert.equal(calls.length, 1);
const { url, init } = calls[0];
assert.equal(url, WORKFLOW_URL);
assert.match(url, /robcam55\/metaculus-bot\/actions\/workflows\/run_bot_on_tournament\.yaml\/dispatches$/);
assert.equal(init.method, "POST");
assert.equal(init.headers.Authorization, "Bearer fake-token");
assert.ok(init.headers["User-Agent"], "GitHub's API rejects requests without a User-Agent");
assert.deepEqual(JSON.parse(init.body), { ref: "main", inputs: { source: "cloudflare" } });

// GitHub refusing (bad token, missing workflow input) fails the tick, so it shows in the logs
globalThis.fetch = async () => new Response("Bad credentials", { status: 401 });
await assert.rejects(worker.scheduled(tick, { GITHUB_TOKEN: "fake-token" }), /401: Bad credentials/);

// No token yet: fail without calling GitHub
calls.length = 0;
globalThis.fetch = async (url, init) => {
  calls.push({ url, init });
  return new Response(null, { status: 204 });
};
await assert.rejects(worker.scheduled(tick, {}), /GITHUB_TOKEN is not set/);
assert.equal(calls.length, 0);

console.log("trigger: ok");
