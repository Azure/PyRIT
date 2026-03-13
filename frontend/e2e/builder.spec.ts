import { expect, test } from "@playwright/test"

import { mockBuilderApis } from "./builderTestUtils"

test.describe("Prompt builder accelerators", () => {
  test("applies a starter, requests multiple versions, and generates a reference image", async ({ page }) => {
    const { buildRequests, referenceImageRequests } = await mockBuilderApis(page)

    await page.goto("/")

    await expect(page.getByRole("heading", { name: "Prompt builder" })).toBeVisible()
    await page.getByRole("button", { name: "Variation", exact: true }).click()

    await page.getByRole("textbox", { name: /Character concept/i }).fill("a masked antihero")
    await page.getByRole("button", { name: /Apply starter to prompt/i }).click()

    await expect(page.getByRole("textbox", { name: /Text to transform/i })).toHaveValue(
      /masked antihero/,
    )

    await page.getByRole("switch").click()
    await expect(page.getByRole("textbox", { name: /Blocked words list/i })).toBeVisible()

    await page.getByRole("combobox", { name: "Versions" }).click()
    await page.getByRole("option", { name: "3" }).click()

    await page.getByRole("button", { name: /Build transformed output/i }).click()

    await expect(page.getByRole("button", { name: "Variation 2" })).toBeVisible()
    await expect(page.getByText("Builder steps")).toBeVisible()
    await expect(page.getByText("Blocked-word avoidance")).toBeVisible()

    await page.getByRole("button", { name: "Variation 3" }).click()
    await page.getByRole("button", { name: /Generate reference image/i }).click()

    await expect(page.getByAltText("Generated reference image")).toBeVisible()
    expect(buildRequests).toHaveLength(1)
    expect(buildRequests[0]?.avoid_blocked_words).toBe(true)
    expect(buildRequests[0]?.variant_count).toBe(3)
    expect(referenceImageRequests).toHaveLength(1)
    expect(referenceImageRequests[0]?.prompt).toBe("third variation")
  })

  test("reuses the latest generated image in an image-based option", async ({ page }) => {
    await mockBuilderApis(page)

    await page.goto("/")
    await page.getByRole("button", { name: "Variation", exact: true }).click()

    await page.getByRole("textbox", { name: /Character concept/i }).fill("a masked antihero")
    await page.getByRole("button", { name: /Apply starter to prompt/i }).click()
    await page.getByRole("button", { name: /Build transformed output/i }).click()
    await page.getByRole("button", { name: /Generate reference image/i }).click()

    await page.getByRole("button", { name: "Add text to image", exact: true }).click()
    await expect(page.getByRole("button", { name: /Use latest generated image/i })).toBeVisible()
    await page.getByRole("button", { name: /Use latest generated image/i }).click()

    await expect(page.getByRole("textbox", { name: /Reference image path or URL/i })).toHaveValue(
      "/tmp/reference-image.png",
    )
  })
})
