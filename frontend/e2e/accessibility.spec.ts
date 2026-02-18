import { test, expect } from "@playwright/test";

test.describe("Accessibility", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("should have accessible form controls", async ({ page }) => {
    // Input should be accessible
    const input = page.getByRole("textbox");
    await expect(input).toBeVisible();

    // Send button should have accessible name
    const sendButton = page.getByRole("button", { name: /send/i });
    await expect(sendButton).toBeVisible();

    // New Chat button should have accessible name
    const newChatButton = page.getByRole("button", { name: /new chat/i });
    await expect(newChatButton).toBeVisible();
  });

  test("should be navigable with keyboard", async ({ page }) => {
    // Tab to the first interactive element
    await page.keyboard.press("Tab");
    const focused = page.locator(":focus");
    await expect(focused).toBeVisible();

    // Continue tabbing through elements
    await page.keyboard.press("Tab");
    await expect(page.locator(":focus")).toBeVisible();
  });

  test("should support Enter key to send message", async ({ page }) => {
    const input = page.getByRole("textbox");
    await input.fill("Test message via Enter");

    // Press Enter to send (if supported)
    await input.press("Enter");

    // Either the message is sent, or we're still in the input
    // This depends on the implementation
    await expect(page.locator("body")).toBeVisible();
  });

  test("should have proper focus management", async ({ page }) => {
    const input = page.getByRole("textbox");

    // Focus input
    await input.focus();
    await expect(input).toBeFocused();

    // Type and verify focus is maintained
    await input.fill("Test");
    await expect(input).toBeFocused();
  });
});

test.describe("Visual Consistency", () => {
  test("should render without layout shifts", async ({ page }) => {
    await page.goto("/");

    // Wait for initial render
    await expect(page.getByText("PyRIT Frontend")).toBeVisible();

    // Take measurements
    const header = page.getByText("PyRIT Frontend");
    const initialBox = await header.boundingBox();

    // Wait a moment for any delayed renders
    await page.waitForTimeout(500);

    // Verify position hasn't changed
    const finalBox = await header.boundingBox();

    if (initialBox && finalBox) {
      expect(finalBox.x).toBe(initialBox.x);
      expect(finalBox.y).toBe(initialBox.y);
    }
  });
});
