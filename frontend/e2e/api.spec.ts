import { test, expect } from "@playwright/test";

// API tests go through the Vite dev server proxy (/api -> backend:8000)
// rather than hitting the backend directly, so they work as soon as
// Playwright's webServer (port 3000) is ready.

test.describe("API Health Check", () => {
  // The backend may still be starting when Vite (port 3000) is already up.
  // Poll the health endpoint through the proxy until the backend is ready.
  test.beforeAll(async ({ request }) => {
    const maxWait = 30_000;
    const interval = 1_000;
    const start = Date.now();
    while (Date.now() - start < maxWait) {
      try {
        const resp = await request.get("/api/health");
        if (resp.ok()) return;
      } catch {
        // Backend not ready yet
      }
      await new Promise((r) => setTimeout(r, interval));
    }
    throw new Error("Backend did not become healthy within 30 seconds");
  });

  test("should have healthy backend API", async ({ request }) => {
    const response = await request.get("/api/health");

    expect(response.ok()).toBe(true);
    const data = await response.json();
    expect(data).toBeDefined();
  });

  test("should get version from API", async ({ request }) => {
    const response = await request.get("/api/version");

    expect(response.ok()).toBe(true);
    const data = await response.json();
    expect(data).toBeDefined();
  });
});

test.describe("Error Handling", () => {
  test("should display UI when backend is slow", async ({ page }) => {
    // Intercept and delay API calls
    await page.route("**/api/**", async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 2000));
      route.continue();
    });

    await page.goto("/");

    // UI should be responsive
    await expect(page.getByRole("textbox")).toBeVisible({ timeout: 10000 });
  });
});
