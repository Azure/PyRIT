import { defineConfig, devices } from "@playwright/test";

const frontendPort = 4173;
const backendPort = 8003;

process.env.PYRIT_FRONTEND_PORT = String(frontendPort);
process.env.PYRIT_BACKEND_PORT = String(backendPort);
process.env.PYRIT_BACKEND_PROXY_URL = `http://127.0.0.1:${backendPort}`;

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: [["html", { open: "never" }], ["list"]],
  timeout: 30000,

  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
    // Firefox can be enabled by installing: npx playwright install firefox
    // {
    //   name: "firefox",
    //   use: { ...devices["Desktop Firefox"] },
    // },
  ],

  /* Automatically start servers before running tests */
  webServer: {
    command: process.env.CI ? "cd .. && uv run python frontend/dev.py" : "../.venv/bin/python dev.py",
    url: `http://127.0.0.1:${frontendPort}`,
    reuseExistingServer: !process.env.CI,
    // CI needs extra time for uv sync + backend startup
    timeout: 120_000,
  },
});
