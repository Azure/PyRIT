import { expect, test } from "@playwright/test"

test.describe("Backend API", () => {
  test("returns health, version, and builder config", async ({ request }) => {
    const healthResponse = await request.get("/api/health")
    expect(healthResponse.ok()).toBe(true)

    const versionResponse = await request.get("/api/version")
    expect(versionResponse.ok()).toBe(true)

    const builderConfigResponse = await request.get("/api/builder/config")
    expect(builderConfigResponse.ok()).toBe(true)

    const builderConfig = await builderConfigResponse.json()
    expect(Array.isArray(builderConfig.families)).toBe(true)
    expect(Array.isArray(builderConfig.presets)).toBe(true)
    expect(Array.isArray(builderConfig.defaults.default_blocked_words)).toBe(true)
  })
})
