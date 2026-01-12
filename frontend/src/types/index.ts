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
