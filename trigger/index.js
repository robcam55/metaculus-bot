// Starts the bot's tournament workflow every 20 minutes. GitHub's own schedule fires this
// repo's 20-minute cron only about five times a day, while FutureEval questions stay open
// for 1.5 to 3 hours. Deploy from the repo root with:
//   npx wrangler deploy --config trigger/wrangler.jsonc
// The token is a fine-grained GitHub token limited to this repo's Actions (read and write):
//   npx wrangler secret put GITHUB_TOKEN --config trigger/wrangler.jsonc

export const WORKFLOW_URL =
  "https://api.github.com/repos/robcam55/metaculus-bot/actions/workflows/run_bot_on_tournament.yaml/dispatches";

export default {
  async scheduled(controller, env) {
    if (!env.GITHUB_TOKEN) {
      throw new Error("GITHUB_TOKEN is not set: run wrangler secret put GITHUB_TOKEN");
    }
    const response = await fetch(WORKFLOW_URL, {
      method: "POST",
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        "User-Agent": "metaculus-bot-trigger",
        "X-GitHub-Api-Version": "2022-11-28",
      },
      // "cloudflare" tells the workflow to honour BOT_ENABLED, as its own schedule does
      body: JSON.stringify({ ref: "main", inputs: { source: "cloudflare" } }),
    });
    if (!response.ok) {
      // A failed tick shows as an error in the Worker's logs; the next tick tries again
      const detail = (await response.text()).slice(0, 300);
      throw new Error(`GitHub answered ${response.status}: ${detail}`);
    }
    console.log(`Started the tournament workflow (${controller.cron})`);
  },
};
