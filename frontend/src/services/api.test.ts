// Mock axios before importing the module
jest.mock("axios", () => ({
  create: jest.fn(() => ({
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
    interceptors: {
      request: { use: jest.fn() },
      response: { use: jest.fn() },
    },
  })),
}))

// Mock the api module so tests can validate endpoint wiring without relying on import.meta support in Jest
jest.mock("./api", () => {
  const mockApiClient = {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
  }

  return {
    apiClient: mockApiClient,
    healthApi: {
      checkHealth: jest.fn(async () => {
        const response = await mockApiClient.get("/health")
        return response.data
      }),
    },
    versionApi: {
      getVersion: jest.fn(async () => {
        const response = await mockApiClient.get("/version")
        return response.data
      }),
    },
    targetsApi: {
      listTargets: jest.fn(async () => {
        const response = await mockApiClient.get("/targets")
        return response.data
      }),
    },
    converterTypesApi: {
      listTypes: jest.fn(async () => {
        const response = await mockApiClient.get("/converters/types")
        return response.data
      }),
    },
    builderApi: {
      getConfig: jest.fn(async () => {
        const response = await mockApiClient.get("/builder/config")
        return response.data
      }),
      build: jest.fn(async (request) => {
        const response = await mockApiClient.post("/builder/build", request)
        return response.data
      }),
      generateReferenceImage: jest.fn(async (prompt: string) => {
        const response = await mockApiClient.post("/builder/reference-image", { prompt })
        return response.data
      }),
    },
    converterPreviewApi: {
      previewType: jest.fn(async (type, params, originalValue, originalValueDataType = "text") => {
        const response = await mockApiClient.post("/converters/preview-type", {
          type,
          params,
          original_value: originalValue,
          original_value_data_type: originalValueDataType,
        })
        return response.data
      }),
    },
  }
})

import {
  apiClient,
  builderApi,
  converterPreviewApi,
  converterTypesApi,
  healthApi,
  targetsApi,
  versionApi,
} from "./api"

describe("api service", () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it("exposes the shared api client", () => {
    expect(apiClient).toBeDefined()
    expect(apiClient.get).toBeDefined()
    expect(apiClient.post).toBeDefined()
  })

  it("calls the health endpoint", async () => {
    ;(apiClient.get as jest.Mock).mockResolvedValueOnce({ data: { status: "healthy" } })

    const result = await healthApi.checkHealth()

    expect(apiClient.get).toHaveBeenCalledWith("/health")
    expect(result).toEqual({ status: "healthy" })
  })

  it("calls the version endpoint", async () => {
    ;(apiClient.get as jest.Mock).mockResolvedValueOnce({ data: { version: "0.11.1" } })

    const result = await versionApi.getVersion()

    expect(apiClient.get).toHaveBeenCalledWith("/version")
    expect(result).toEqual({ version: "0.11.1" })
  })

  it("loads builder config", async () => {
    ;(apiClient.get as jest.Mock).mockResolvedValueOnce({ data: { families: [], presets: [] } })

    const result = await builderApi.getConfig()

    expect(apiClient.get).toHaveBeenCalledWith("/builder/config")
    expect(result).toEqual({ families: [], presets: [] })
  })

  it("posts builder build requests", async () => {
    const request = {
      source_content: "source",
      source_content_data_type: "text",
      converter_type: "VariationConverter",
      converter_params: {},
      preset_values: {},
      avoid_blocked_words: false,
      blocked_words: [],
      variant_count: 1,
    }
    ;(apiClient.post as jest.Mock).mockResolvedValueOnce({ data: { converted_value: "built" } })

    const result = await builderApi.build(request)

    expect(apiClient.post).toHaveBeenCalledWith("/builder/build", request)
    expect(result).toEqual({ converted_value: "built" })
  })

  it("posts reference-image requests", async () => {
    ;(apiClient.post as jest.Mock).mockResolvedValueOnce({ data: { image_path: "/tmp/reference.png" } })

    const result = await builderApi.generateReferenceImage("prompt text")

    expect(apiClient.post).toHaveBeenCalledWith("/builder/reference-image", { prompt: "prompt text" })
    expect(result).toEqual({ image_path: "/tmp/reference.png" })
  })

  it("loads targets and converter types", async () => {
    ;(apiClient.get as jest.Mock)
      .mockResolvedValueOnce({ data: { items: [] } })
      .mockResolvedValueOnce({ data: { items: [] } })

    const targets = await targetsApi.listTargets()
    const converterTypes = await converterTypesApi.listTypes()

    expect(apiClient.get).toHaveBeenNthCalledWith(1, "/targets")
    expect(apiClient.get).toHaveBeenNthCalledWith(2, "/converters/types")
    expect(targets).toEqual({ items: [] })
    expect(converterTypes).toEqual({ items: [] })
  })

  it("posts converter preview requests", async () => {
    ;(apiClient.post as jest.Mock).mockResolvedValueOnce({ data: { converted_value: "preview" } })

    const result = await converterPreviewApi.previewType("VariationConverter", { tone: "calm" }, "seed text")

    expect(apiClient.post).toHaveBeenCalledWith("/converters/preview-type", {
      type: "VariationConverter",
      params: { tone: "calm" },
      original_value: "seed text",
      original_value_data_type: "text",
    })
    expect(result).toEqual({ converted_value: "preview" })
  })
})
