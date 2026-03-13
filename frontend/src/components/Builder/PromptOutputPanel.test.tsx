import { fireEvent, render, screen } from "@testing-library/react"
import PromptOutputPanel from "./PromptOutputPanel"
import type { BuilderBuildResponse, ConverterTypeMetadata } from "../../types"

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

const buildResponse: BuilderBuildResponse = {
  resolved_source_value: "rewritten source prompt",
  resolved_source_data_type: "text",
  converted_value: "base version",
  converted_value_data_type: "text",
  variants: [
    {
      variant_id: "base",
      label: "Base version",
      value: "base version",
      data_type: "text",
      kind: "base",
    },
    {
      variant_id: "variation-2",
      label: "Variation 2",
      value: "variation version",
      data_type: "text",
      kind: "variation",
    },
  ],
  steps: [
    {
      stage: "blocked_words",
      title: "Blocked-word avoidance",
      input_value: "source prompt",
      input_data_type: "text",
      output_value: "rewritten source prompt",
      output_data_type: "text",
      detail: "Rephrased obvious trigger words.",
    },
    {
      stage: "converter",
      title: "Converter: VariationConverter",
      input_value: "rewritten source prompt",
      input_data_type: "text",
      output_value: "base version",
      output_data_type: "text",
      detail: "Applied the selected converter.",
    },
  ],
  warnings: ["Image helper is not configured."],
}

describe("PromptOutputPanel", () => {
  it("shows variants, builder steps, and unavailable image guidance", () => {
    const onSelectVariant = jest.fn()

    render(
      <PromptOutputPanel
        option={option}
        basePrompt="raw source prompt"
        isPromptReady={true}
        buildResponse={buildResponse}
        selectedVariantId="base"
        isPreviewLoading={false}
        previewError={null}
        canPreview={true}
        previewHint="Build the prompt."
        onBuildTransformedPrompt={jest.fn()}
        onSelectVariant={onSelectVariant}
        onCopy={jest.fn()}
        referenceImage={null}
        referenceImageAvailable={false}
        referenceImageTargetName={null}
        isReferenceImageLoading={false}
        referenceImageError={null}
        onGenerateReferenceImage={jest.fn()}
      />,
    )

    expect(screen.getAllByText("rewritten source prompt")).toHaveLength(2)
    expect(screen.getByText("Builder steps")).toBeInTheDocument()
    expect(screen.getByText("Blocked-word avoidance")).toBeInTheDocument()
    expect(screen.getByText("Image helper is not configured.")).toBeInTheDocument()
    expect(screen.getByText(/Reference image generation is unavailable/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Variation 2" }))
    expect(onSelectVariant).toHaveBeenCalledWith("variation-2")
  })
})
