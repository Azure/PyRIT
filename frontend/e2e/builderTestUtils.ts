import type { Page } from "@playwright/test"

export async function mockBuilderApis(page: Page) {
  const buildRequests: Record<string, unknown>[] = []
  const referenceImageRequests: Record<string, unknown>[] = []

  await page.route("**/api/converters/types", async (route) => {
    await route.fulfill({
      json: {
        items: [
          {
            converter_type: "VariationConverter",
            display_name: "Variation",
            description: "Generate prompt variants",
            supported_input_types: ["text"],
            supported_output_types: ["text"],
            parameters: [],
            preview_supported: true,
            preview_unavailable_reason: null,
          },
          {
            converter_type: "AddTextImageConverter",
            display_name: "Add text to image",
            description: "Apply an image-based flow",
            supported_input_types: ["image_path"],
            supported_output_types: ["image_path"],
            parameters: [],
            preview_supported: true,
            preview_unavailable_reason: null,
          },
        ],
      },
    })
  })

  await page.route("**/api/targets", async (route) => {
    await route.fulfill({
      json: {
        items: [],
        pagination: {
          limit: 20,
          has_more: false,
          next_cursor: null,
          prev_cursor: null,
        },
      },
    })
  })

  await page.route("**/api/builder/config", async (route) => {
    await route.fulfill({
      json: {
        families: [
          {
            family_id: "single_turn",
            title: "Single-turn starters",
            summary: "Quick single-turn seeds",
            preset_ids: ["single_roleplay_director"],
          },
        ],
        presets: [
          {
            preset_id: "single_roleplay_director",
            family_id: "single_turn",
            title: "Role-play director",
            summary: "Frames the user as a director.",
            template: "Create a tense video prompt for {{ character_concept }} in {{ scene_setting }} with {{ goal }}.",
            fields: [
              {
                name: "character_concept",
                label: "Character concept",
                placeholder: "A character idea",
                required: true,
                default_value: "",
              },
              {
                name: "scene_setting",
                label: "Scene or setting",
                placeholder: "A location",
                required: false,
                default_value: "a rain-soaked alley",
              },
              {
                name: "goal",
                label: "Goal",
                placeholder: "Attack goal",
                required: false,
                default_value: "escalating pressure",
              },
            ],
          },
        ],
        defaults: {
          default_blocked_words: ["kill", "gore", "bomb"],
          max_variant_count: 5,
          multi_variant_converter_types: ["VariationConverter"],
        },
        capabilities: {
          reference_image_available: true,
          reference_image_target_name: "image-helper",
        },
      },
    })
  })

  await page.route("**/api/builder/build", async (route) => {
    const payload = route.request().postDataJSON() as Record<string, unknown>
    buildRequests.push(payload)

    await route.fulfill({
      json: {
        resolved_source_value: payload.source_content || "Create a tense video prompt",
        resolved_source_data_type: "text",
        converted_value: "base variation",
        converted_value_data_type: "text",
        variants: [
          {
            variant_id: "base",
            label: "Base version",
            value: "base variation",
            data_type: "text",
            kind: "base",
          },
          {
            variant_id: "variation-2",
            label: "Variation 2",
            value: "second variation",
            data_type: "text",
            kind: "variation",
          },
          {
            variant_id: "variation-3",
            label: "Variation 3",
            value: "third variation",
            data_type: "text",
            kind: "variation",
          },
        ],
        steps: [
          {
            stage: "preset",
            title: "Preset: Role-play director",
            input_value: "",
            input_data_type: "text",
            output_value: payload.source_content || "Create a tense video prompt",
            output_data_type: "text",
            detail: "Preset fields were resolved into the working prompt.",
          },
          {
            stage: "blocked_words",
            title: "Blocked-word avoidance",
            input_value: payload.source_content || "Create a tense video prompt",
            input_data_type: "text",
            output_value: "rewritten source",
            output_data_type: "text",
            detail: "Rephrased obvious trigger words.",
          },
          {
            stage: "converter",
            title: "Converter: VariationConverter",
            input_value: "rewritten source",
            input_data_type: "text",
            output_value: "base variation",
            output_data_type: "text",
            detail: "Applied the selected converter.",
          },
          {
            stage: "variants",
            title: "Additional versions",
            input_value: "base variation",
            input_data_type: "text",
            output_value: "second variation\n\nthird variation",
            output_data_type: "text",
            detail: "Generated alternate versions of the built prompt.",
          },
        ],
        warnings: [],
      },
    })
  })

  await page.route("**/api/builder/reference-image", async (route) => {
    const payload = route.request().postDataJSON() as Record<string, unknown>
    referenceImageRequests.push(payload)

    await route.fulfill({
      json: {
        prompt: payload.prompt,
        image_path: "/tmp/reference-image.png",
        image_url: "/api/media?path=%2Ftmp%2Freference-image.png",
        data_type: "image_path",
        target_name: "image-helper",
      },
    })
  })

  return {
    buildRequests,
    referenceImageRequests,
  }
}
