import axios from 'axios'
import { ChatRequest, ChatResponse, ConversationHistory, TargetInfo } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const chatApi = {
  sendMessage: async (request: ChatRequest): Promise<ChatResponse> => {
    // If there are attachments with files, use multipart/form-data
    if (request.attachments && request.attachments.length > 0 && request.attachments.some(a => a.file)) {
      const formData = new FormData()
      formData.append('message', request.message)
      if (request.conversation_id) {
        formData.append('conversation_id', request.conversation_id)
      }
      if (request.target_id) {
        formData.append('target_id', request.target_id)
      }
      
      // Add all files
      request.attachments.forEach((attachment) => {
        if (attachment.file) {
          formData.append('files', attachment.file)
        }
      })
      
      const response = await apiClient.post<ChatResponse>('/chat', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      return response.data
    } else {
      // Regular JSON request
      const response = await apiClient.post<ChatResponse>('/chat', request)
      return response.data
    }
  },

  getConversations: async (): Promise<ConversationHistory[]> => {
    const response = await apiClient.get<ConversationHistory[]>('/chat/conversations')
    return response.data
  },

  getConversation: async (conversationId: string): Promise<ConversationHistory> => {
    const response = await apiClient.get<ConversationHistory>(`/chat/conversations/${conversationId}`)
    return response.data
  },

  deleteConversation: async (conversationId: string): Promise<void> => {
    await apiClient.delete(`/chat/conversations/${conversationId}`)
  },
}

export const targetsApi = {
  listTargets: async (): Promise<TargetInfo[]> => {
    const response = await apiClient.get<TargetInfo[]>('/targets')
    return response.data
  },

  getTarget: async (targetId: string): Promise<TargetInfo> => {
    const response = await apiClient.get<TargetInfo>(`/targets/${targetId}`)
    return response.data
  },
}

export const healthApi = {
  checkHealth: async () => {
    const response = await apiClient.get('/health')
    return response.data
  },
  getVersion: async () => {
    const response = await apiClient.get('/version')
    return response.data
  },
}

export const configApi = {
  getEnvVars: async () => {
    const response = await apiClient.get('/config/env-vars')
    return response.data
  },
  getEnvVarValue: async (varName: string) => {
    const response = await apiClient.get(`/config/env-vars/${varName}`)
    return response.data
  },
  getTargetTypes: async () => {
    const response = await apiClient.get('/config/target-types')
    return response.data
  },
}
