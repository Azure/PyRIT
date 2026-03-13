import { fireEvent, render, screen } from "@testing-library/react"
import AttackStarterPanel from "./AttackStarterPanel"
import type {
  BuilderConfigResponse,
  ConverterTypeMetadata,
  PromptBankPreset,
  PromptBuilderFormState,
  ReferenceImageResponse,
} from "../../types"

const option: ConverterTypeMetadata = {
  converter_type: "VariationConverter",
  display_name: "Variation",
  description: "Generate prompt variants",
  supported_input_types: ["text"],
  supported_output_types: ["text"],
  parameters: [],
  preview_supported: true,
  preview_unavailable_reason: null,
}

const preset: PromptBankPreset = {
  preset_id: "single_roleplay_director",
  family_id: "single_turn",
  title: "Role-play director",
  summary: "Frames the user as a director.",
  template: "Create a prompt for {{ character_concept }}.",
  fields: [
    {
      name: "character_concept",
      label: "Character concept",
      placeholder: "A character idea",
      required: true,
      default_value: "",
    },
  ],
}

const config: BuilderConfigResponse = {
  families: [
    {
      family_id: "single_turn",
      title: "Single-turn starters",
      summary: "Quick single-turn seeds",
      preset_ids: [preset.preset_id],
    },
  ],
  presets: [preset],
  defaults: {
    default_blocked_words: ["kill", "gore"],
    max_variant_count: 5,
    multi_variant_converter_types: ["VariationConverter"],
  },
  capabilities: {
    reference_image_available: true,
    reference_image_target_name: "image-helper",
  },
}

const formState: PromptBuilderFormState = {
  selectedTargetId: "",
  sourceContent: "",
  selectedPresetId: preset.preset_id,
  presetValues: {
    character_concept: "masked antihero",
  },
  avoidBlockedWords: true,
  blockedWordsText: "kill\ngore",
  variantCount: 3,
  parameterValues: {},
}

const latestGeneratedImage: ReferenceImageResponse = {
  prompt: "prompt",
  image_path: "/tmp/reference.png",
  image_url: "/api/media?path=%2Ftmp%2Freference.png",
  data_type: "image_path",
  target_name: "image-helper",
}

describe("AttackStarterPanel", () => {
  it("shows blocked-word controls and starter actions for text options", () => {
    const onAvoidBlockedWordsChange = jest.fn()

    render(
      <AttackStarterPanel
        option={option}
        config={config}
        formState={formState}
        selectedFamilyId="single_turn"
        selectedPreset={preset}
        latestGeneratedImage={latestGeneratedImage}
        onFamilyChange={jest.fn()}
        onPresetChange={jest.fn()}
        onPresetFieldChange={jest.fn()}
        onApplyPreset={jest.fn()}
        onAvoidBlockedWordsChange={onAvoidBlockedWordsChange}
        onBlockedWordsTextChange={jest.fn()}
        onVariantCountChange={jest.fn()}
        onUseLatestGeneratedImage={jest.fn()}
      />,
    )

    expect(screen.getByText("Attack starter")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /Apply starter to prompt/i })).toBeInTheDocument()
    expect(screen.getByRole("textbox", { name: /Blocked words list/i })).toHaveValue("kill\ngore")
    expect(screen.getByRole("combobox", { name: "Versions" })).toBeInTheDocument()

    fireEvent.click(screen.getByRole("switch", { name: /Avoid obviously blocked words/i }))
    expect(onAvoidBlockedWordsChange).toHaveBeenCalledWith(false)
  })
})
