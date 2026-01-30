import { test, expect } from "@playwright/test";

test.describe("API Health Check", () => {
  test("should have healthy backend API", async ({ request }) => {
    const response = await request.get("http://localhost:8000/api/health");

    expect(response.ok()).toBe(true);
    const data = await response.json();
    expect(data).toBeDefined();
  });

  test("should get version from API", async ({ request }) => {
    const response = await request.get("http://localhost:8000/api/version");

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
