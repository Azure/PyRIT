import { expect, test } from "@playwright/test"

import { mockBuilderApis } from "./builderTestUtils"

test.describe("Prompt builder accessibility", () => {
  test("exposes labeled builder controls", async ({ page }) => {
    await mockBuilderApis(page)
    await page.goto("/")

    await expect(page.getByRole("heading", { name: "Prompt builder" })).toBeVisible()
    await expect(page.getByRole("heading", { name: "Attack starter" })).toBeVisible()
    await expect(page.getByRole("heading", { name: "Attack options" })).toBeVisible()
    await expect(page.getByRole("heading", { name: "Built output" })).toBeVisible()

    await page.getByRole("button", { name: "Variation", exact: true }).click()

    await expect(page.getByRole("combobox", { name: "Attack family" })).toBeVisible()
    await expect(page.getByRole("combobox", { name: "Starter preset" })).toBeVisible()
    await expect(page.getByRole("combobox", { name: "Versions" })).toBeVisible()
    await expect(page.getByRole("textbox", { name: /Text to transform/i })).toBeVisible()
  })

  test("supports keyboard movement through the builder controls", async ({ page }) => {
    await mockBuilderApis(page)
    await page.goto("/")

    await page.keyboard.press("Tab")
    await expect(page.locator(":focus")).toBeVisible()

    await page.keyboard.press("Tab")
    await expect(page.locator(":focus")).toBeVisible()

    await page.getByRole("button", { name: "Variation", exact: true }).click()
    await page.getByRole("textbox", { name: /Character concept/i }).focus()
    await expect(page.getByRole("textbox", { name: /Character concept/i })).toBeFocused()

    await page.keyboard.type("masked antihero")
    await expect(page.getByRole("textbox", { name: /Character concept/i })).toHaveValue("masked antihero")
  })
})
