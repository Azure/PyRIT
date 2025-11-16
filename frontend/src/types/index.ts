export interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
}

export interface ChatRequest {
  message: string
  conversation_id?: string
  target_id?: string
}

export interface ChatResponse {
  conversation_id: string
  message: string
  role: string
  timestamp: string
  target_id?: string
}

export interface ConversationHistory {
  conversation_id: string
  messages: Message[]
  created_at: string
  updated_at: string
  target_id?: string
}

export interface TargetInfo {
  id: string
  name: string
  type: string
  description: string
  status: string
}
