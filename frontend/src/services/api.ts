import axios from 'axios'
import { TargetInfo } from '../types'

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export { apiClient }

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

