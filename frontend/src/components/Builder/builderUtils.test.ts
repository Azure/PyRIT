import {
  canRequestVariants,
  expandPresetTemplate,
  formatBlockedWords,
  getInitialPresetValues,
  parseBlockedWords,
} from "./builderUtils"
import type { BuilderConfigResponse, PromptBankPreset } from "../../types"

const preset: PromptBankPreset = {
  preset_id: "single_roleplay_director",
  family_id: "single_turn",
  title: "Role-play director",
  summary: "Frames the request like exacting scene direction.",
  template: "Create a prompt for {{ character_concept }} in {{ scene_setting }} with {{ goal }}.",
  fields: [
    {
      name: "character_concept",
      label: "Character concept",
      required: true,
      default_value: "",
    },
    {
      name: "scene_setting",
      label: "Scene",
      required: false,
      default_value: "a rehearsal studio",
    },
    {
      name: "goal",
      label: "Goal",
      required: false,
      default_value: "cinematic tension",
    },
  ],
}

const builderConfig: BuilderConfigResponse = {
  families: [],
  presets: [preset],
  defaults: {
    default_blocked_words: ["kill", "gore"],
    max_variant_count: 5,
    multi_variant_converter_types: ["VariationConverter"],
  },
  capabilities: {
    reference_image_available: false,
    reference_image_target_name: null,
  },
}

describe("builder utils", () => {
  it("expands preset templates with field values and trims the result", () => {
    expect(
      expandPresetTemplate(preset, {
        character_concept: "a masked antihero",
        scene_setting: "a rain-soaked alley",
        goal: "tense pursuit energy",
      }),
    ).toBe("Create a prompt for a masked antihero in a rain-soaked alley with tense pursuit energy.")
  })

  it("returns default preset field values", () => {
    expect(getInitialPresetValues(preset)).toEqual({
      character_concept: "",
      scene_setting: "a rehearsal studio",
      goal: "cinematic tension",
    })
  })

  it("parses and formats blocked-word lists", () => {
    const parsed = parseBlockedWords("kill\ngore, bomb , \n")

    expect(parsed).toEqual(["kill", "gore", "bomb"])
    expect(formatBlockedWords(parsed)).toBe("kill\ngore\nbomb")
  })

  it("only enables multi-version support for configured converter types", () => {
    expect(
      canRequestVariants(
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
        builderConfig,
      ),
    ).toBe(true)

    expect(
      canRequestVariants(
        {
          converter_type: "AddTextImageConverter",
          display_name: "Add text to image",
          description: "Uses an image input",
          supported_input_types: ["image_path"],
          supported_output_types: ["image_path"],
          parameters: [],
          preview_supported: true,
          preview_unavailable_reason: null,
        },
        builderConfig,
      ),
    ).toBe(false)
  })
})
