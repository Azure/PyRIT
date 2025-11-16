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
    const response = await apiClient.post<ChatResponse>('/chat', request)
    return response.data
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
  checkHealth: async (): Promise<{ status: string; timestamp: string; service: string }> => {
    const response = await apiClient.get('/health')
    return response.data
  },

  getVersion: async (): Promise<{ version: string; api_version: string }> => {
    const response = await apiClient.get('/version')
    return response.data
  },
}
