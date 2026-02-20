import axios from 'axios'
import type {
  TargetInstance,
  TargetListResponse,
  CreateTargetRequest,
  CreateAttackRequest,
  CreateAttackResponse,
  AttackSummary,
  AttackListResponse,
  AttackMessagesResponse,
  AddMessageRequest,
  AddMessageResponse,
} from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export { apiClient }

export const healthApi = {
  checkHealth: async () => {
    const response = await apiClient.get('/health')
    return response.data
  },
}

export const versionApi = {
  getVersion: async () => {
    const response = await apiClient.get('/version')
    return response.data
  },
}

export const targetsApi = {
  listTargets: async (limit = 50, cursor?: string): Promise<TargetListResponse> => {
    const params: Record<string, string | number> = { limit }
    if (cursor) params.cursor = cursor
    const response = await apiClient.get('/targets', { params })
    return response.data
  },

  getTarget: async (targetRegistryName: string): Promise<TargetInstance> => {
    const response = await apiClient.get(`/targets/${encodeURIComponent(targetRegistryName)}`)
    return response.data
  },

  createTarget: async (request: CreateTargetRequest): Promise<TargetInstance> => {
    const response = await apiClient.post('/targets', request)
    return response.data
  },
}

export const attacksApi = {
  createAttack: async (request: CreateAttackRequest): Promise<CreateAttackResponse> => {
    const response = await apiClient.post('/attacks', request)
    return response.data
  },

  getAttack: async (conversationId: string): Promise<AttackSummary> => {
    const response = await apiClient.get(`/attacks/${encodeURIComponent(conversationId)}`)
    return response.data
  },

  getMessages: async (conversationId: string): Promise<AttackMessagesResponse> => {
    const response = await apiClient.get(`/attacks/${encodeURIComponent(conversationId)}/messages`)
    return response.data
  },

  addMessage: async (conversationId: string, request: AddMessageRequest): Promise<AddMessageResponse> => {
    const response = await apiClient.post(
      `/attacks/${encodeURIComponent(conversationId)}/messages`,
      request
    )
    return response.data
  },

  listAttacks: async (params?: {
    limit?: number
    cursor?: string
    attack_class?: string
    outcome?: string
  }): Promise<AttackListResponse> => {
    const response = await apiClient.get('/attacks', { params })
    return response.data
  },
}
