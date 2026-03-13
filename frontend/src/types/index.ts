export interface MessageAttachment {
  type: 'image' | 'audio' | 'video' | 'file'
  name: string
  url: string
  mimeType: string
  size: number
  file?: File
}

export interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
  attachments?: MessageAttachment[]
}

export interface VersionInfo {
  version: string
  display?: string
}

export interface BuilderPresetField {
  name: string
  label: string
  placeholder?: string | null
  required: boolean
  default_value?: string | null
}

export interface PromptBankPreset {
  preset_id: string
  family_id: string
  title: string
  summary: string
  template: string
  fields: BuilderPresetField[]
}

export interface PromptBankFamily {
  family_id: string
  title: string
  summary: string
  preset_ids: string[]
}

export interface BuilderConfigResponse {
  families: PromptBankFamily[]
  presets: PromptBankPreset[]
  defaults: {
    default_blocked_words: string[]
    max_variant_count: number
    multi_variant_converter_types: string[]
  }
  capabilities: {
    reference_image_available: boolean
    reference_image_target_name?: string | null
  }
}

export interface BuilderBuildStep {
  stage: 'preset' | 'blocked_words' | 'converter' | 'variants'
  title: string
  input_value: string
  input_data_type: string
  output_value: string
  output_data_type: string
  detail?: string | null
}

export interface BuilderVariant {
  variant_id: string
  label: string
  value: string
  data_type: string
  kind: 'base' | 'variation'
}

export interface BuilderBuildRequest {
  source_content: string
  source_content_data_type: string
  converter_type: string
  converter_params: Record<string, unknown>
  preset_id?: string | null
  preset_values: Record<string, string>
  avoid_blocked_words: boolean
  blocked_words: string[]
  variant_count: number
}

export interface BuilderBuildResponse {
  resolved_source_value: string
  resolved_source_data_type: string
  converted_value: string
  converted_value_data_type: string
  variants: BuilderVariant[]
  steps: BuilderBuildStep[]
  warnings: string[]
}

export interface ReferenceImageResponse {
  prompt: string
  image_path: string
  image_url: string
  data_type: 'image_path'
  target_name?: string | null
}

export interface TargetInstance {
  target_registry_name: string
  target_type: string
  endpoint?: string | null
  model_name?: string | null
  temperature?: number | null
  top_p?: number | null
  max_requests_per_minute?: number | null
  supports_multi_turn: boolean
  target_specific_params?: Record<string, unknown> | null
}

export interface TargetListResponse {
  items: TargetInstance[]
  pagination: {
    limit: number
    has_more: boolean
    next_cursor?: string | null
    prev_cursor?: string | null
  }
}

export type BuilderInputKind =
  | 'text'
  | 'number'
  | 'boolean'
  | 'select'
  | 'list'
  | 'unsupported'

export interface ConverterParameterMetadata {
  name: string
  display_name: string
  type_label: string
  required: boolean
  default_value?: string | null
  input_kind: BuilderInputKind
  options?: string[] | null
}

export interface ConverterTypeMetadata {
  converter_type: string
  display_name: string
  description: string
  supported_input_types: string[]
  supported_output_types: string[]
  parameters: ConverterParameterMetadata[]
  preview_supported: boolean
  preview_unavailable_reason?: string | null
}

export interface ConverterTypeListResponse {
  items: ConverterTypeMetadata[]
}

export interface ConverterPreviewResponse {
  original_value: string
  original_value_data_type: string
  converted_value: string
  converted_value_data_type: string
  steps: Array<{
    converter_id: string
    converter_type: string
    input_value: string
    input_data_type: string
    output_value: string
    output_data_type: string
  }>
}

export interface PromptBuilderFormState {
  selectedTargetId: string
  sourceContent: string
  selectedPresetId: string
  presetValues: Record<string, string>
  avoidBlockedWords: boolean
  blockedWordsText: string
  variantCount: number
  parameterValues: Record<string, string | number | boolean>
}
