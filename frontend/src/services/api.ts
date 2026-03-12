import axios from 'axios'
import type {
  ConverterPreviewResponse,
  ConverterTypeListResponse,
  TargetListResponse,
  VersionInfo,
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
    const response = await apiClient.get<VersionInfo>('/version')
    return response.data
  },
}

export const targetsApi = {
  listTargets: async () => {
    const response = await apiClient.get<TargetListResponse>('/targets')
    return response.data
  },
}

export const converterTypesApi = {
  listTypes: async () => {
    const response = await apiClient.get<ConverterTypeListResponse>('/converters/types')
    return response.data
  },
}

export const converterPreviewApi = {
  previewType: async (
    type: string,
    params: Record<string, unknown>,
    originalValue: string,
    originalValueDataType = 'text',
  ) => {
    const response = await apiClient.post<ConverterPreviewResponse>('/converters/preview-type', {
      type,
      params,
      original_value: originalValue,
      original_value_data_type: originalValueDataType,
    })
    return response.data
  },
}
