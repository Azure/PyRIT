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
  parameterValues: Record<string, string | number | boolean>
}
