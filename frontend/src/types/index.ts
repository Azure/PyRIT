// ============================================================================
// Frontend UI Types
// ============================================================================

export interface MessageAttachment {
  type: 'image' | 'audio' | 'video' | 'file'
  name: string
  url: string
  mimeType: string
  size: number
  file?: File
}

export interface Message {
  role: 'user' | 'assistant' | 'simulated_assistant' | 'system'
  content: string
  timestamp: string
  attachments?: MessageAttachment[]
  /** If the backend returned an error for this message */
  error?: MessageError
  /** True while waiting for the backend response */
  isLoading?: boolean
}

export interface MessageError {
  type: string // e.g. 'blocked', 'processing', 'empty', 'unknown'
  description?: string
}

// ============================================================================
// Backend DTO Types (mirror pyrit/backend/models)
// ============================================================================

export interface PaginationInfo {
  limit: number
  has_more: boolean
  next_cursor?: string | null
  prev_cursor?: string | null
}

// --- Targets ---

export interface TargetInstance {
  target_registry_name: string
  target_type: string
  endpoint?: string | null
  model_name?: string | null
  temperature?: number | null
  top_p?: number | null
  max_requests_per_minute?: number | null
  target_specific_params?: Record<string, unknown> | null
}

export interface TargetListResponse {
  items: TargetInstance[]
  pagination: PaginationInfo
}

export interface CreateTargetRequest {
  type: string
  params: Record<string, unknown>
}

// --- Attacks ---

export interface TargetInfo {
  target_type: string
  endpoint?: string | null
  model_name?: string | null
}

export interface AttackSummary {
  conversation_id: string
  attack_type: string
  attack_specific_params?: Record<string, unknown> | null
  target?: TargetInfo | null
  converters: string[]
  outcome?: 'undetermined' | 'success' | 'failure' | null
  last_message_preview?: string | null
  message_count: number
  labels: Record<string, string>
  created_at: string
  updated_at: string
}

export interface CreateAttackRequest {
  target_registry_name: string
  name?: string
  labels?: Record<string, string>
}

export interface CreateAttackResponse {
  conversation_id: string
  created_at: string
}

// --- Messages ---

export interface BackendScore {
  score_id: string
  scorer_type: string
  score_value: number
  score_rationale?: string | null
  scored_at: string
}

export interface BackendMessagePiece {
  piece_id: string
  original_value_data_type: string
  converted_value_data_type: string
  original_value?: string | null
  original_value_mime_type?: string | null
  converted_value: string
  converted_value_mime_type?: string | null
  scores: BackendScore[]
  response_error: string // 'none' | 'blocked' | 'processing' | 'empty' | 'unknown'
  response_error_description?: string | null
}

export interface BackendMessage {
  turn_number: number
  role: string
  pieces: BackendMessagePiece[]
  created_at: string
}

export interface AttackMessagesResponse {
  conversation_id: string
  messages: BackendMessage[]
}

export interface MessagePieceRequest {
  data_type: string // 'text' | 'image_path' | 'audio_path' | 'video_path' | 'binary_path'
  original_value: string
  converted_value?: string
  mime_type?: string
}

export interface AddMessageRequest {
  role: string
  pieces: MessagePieceRequest[]
  send: boolean
  target_registry_name?: string
  converter_ids?: string[]
}

export interface AddMessageResponse {
  attack: AttackSummary
  messages: AttackMessagesResponse
}

export interface AttackListResponse {
  items: AttackSummary[]
  pagination: PaginationInfo
}
