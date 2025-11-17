export interface MessageAttachment {
  type: 'image' | 'audio' | 'video' | 'file'
  name: string
  url: string
  mimeType: string
  size: number
  file?: File  // Keep reference to original file for upload
}

export interface Message {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: string
  attachments?: MessageAttachment[]
}

export interface ChatRequest {
  original_value: string  // The original text before any conversion
  converted_value?: string  // The text after conversion (if converters were applied)
  conversation_id?: string
  target_id?: string
  attachments?: MessageAttachment[]
  converter_identifiers?: Array<Record<string, string>>  // PyRIT converter identifiers from preview
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

export interface ConverterParameter {
  name: string;
  type: 'bool' | 'int' | 'float' | 'str' | 'enum';
  required: boolean;
  description?: string;
  default?: any;
  enum_values?: string[];
}

export interface ConverterInfo {
  name: string
  class_name: string
  description?: string
  parameters: ConverterParameter[]
  uses_llm: boolean
}

export interface ConverterInstance {
  id: string
  class_name: string
  config: Record<string, any>
}
