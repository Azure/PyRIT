import { test, expect } from "@playwright/test";

test.describe("Application Smoke Tests", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("should load the application", async ({ page }) => {
    // Wait for the app to load
    await expect(page.locator("body")).toBeVisible();
  });

  test("should display PyRIT Frontend header", async ({ page }) => {
    await expect(page.getByText("PyRIT Frontend")).toBeVisible({ timeout: 10000 });
  });

  test("should have New Chat button", async ({ page }) => {
    await expect(page.getByRole("button", { name: /new chat/i })).toBeVisible();
  });

  test("should have message input", async ({ page }) => {
    await expect(page.getByRole("textbox")).toBeVisible();
  });
});

test.describe("Chat Functionality", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
  });

  test("should send a message and receive echo response", async ({ page }) => {
    const input = page.getByRole("textbox");
    await expect(input).toBeVisible();

    // Type and send message
    await input.fill("Hello, this is a test message");
    await page.getByRole("button", { name: /send/i }).click();

    // Verify user message appears
    await expect(page.getByText("Hello, this is a test message")).toBeVisible();

    // Verify echo response appears
    await expect(page.getByText(/Echo: Hello, this is a test message/)).toBeVisible({ timeout: 5000 });
  });

  test("should clear input after sending", async ({ page }) => {
    const input = page.getByRole("textbox");
    await input.fill("Test message");
    await page.getByRole("button", { name: /send/i }).click();

    // Input should be cleared
    await expect(input).toHaveValue("");
  });

  test("should disable send button when input is empty", async ({ page }) => {
    const sendButton = page.getByRole("button", { name: /send/i });
    await expect(sendButton).toBeDisabled();
  });

  test("should enable send button when input has text", async ({ page }) => {
    const input = page.getByRole("textbox");
    await input.fill("Some text");

    const sendButton = page.getByRole("button", { name: /send/i });
    await expect(sendButton).toBeEnabled();
  });

  test("should start new chat when clicking New Chat", async ({ page }) => {
    // Send a message first
    const input = page.getByRole("textbox");
    await input.fill("First message");
    await page.getByRole("button", { name: /send/i }).click();
    await expect(page.getByText("First message")).toBeVisible();
    await expect(page.getByText(/Echo: First message/)).toBeVisible({ timeout: 5000 });

    // Click New Chat
    await page.getByRole("button", { name: /new chat/i }).click();

    // Previous messages should be cleared
    await expect(page.getByText("First message")).not.toBeVisible();
    await expect(page.getByText(/Echo: First message/)).not.toBeVisible();
  });
});

test.describe("Multiple Messages", () => {
  test("should maintain conversation history", async ({ page }) => {
    await page.goto("/");

    const input = page.getByRole("textbox");

    // Send first message
    await input.fill("First message");
    await page.getByRole("button", { name: /send/i }).click();
    await expect(page.getByText("First message")).toBeVisible();
    await expect(page.getByText(/Echo: First message/)).toBeVisible({ timeout: 5000 });

    // Send second message
    await input.fill("Second message");
    await page.getByRole("button", { name: /send/i }).click();
    await expect(page.getByText("Second message")).toBeVisible();
    await expect(page.getByText(/Echo: Second message/)).toBeVisible({ timeout: 5000 });

    // Both messages should still be visible (use exact match to avoid matching Echo responses)
    await expect(page.getByText("First message", { exact: true })).toBeVisible();
    await expect(page.getByText("Second message", { exact: true })).toBeVisible();
  });
});
