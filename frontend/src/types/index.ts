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

export interface TargetInfo {
  id: string
  name: string
  type: string
  description: string
  status: string
}
