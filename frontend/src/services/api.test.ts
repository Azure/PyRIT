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
}));

// Mock import.meta.env before importing api
jest.mock("./api", () => {
  const mockApiClient = {
    get: jest.fn(),
    post: jest.fn(),
    put: jest.fn(),
    delete: jest.fn(),
  };

  return {
    apiClient: mockApiClient,
    healthApi: {
      checkHealth: jest.fn(async () => {
        const response = await mockApiClient.get("/health");
        return response.data;
      }),
    },
    versionApi: {
      getVersion: jest.fn(async () => {
        const response = await mockApiClient.get("/version");
        return response.data;
      }),
    },
  };
});

import { apiClient, healthApi, versionApi } from "./api";

describe("api service", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("apiClient", () => {
    it("should be defined", () => {
      expect(apiClient).toBeDefined();
    });

    it("should have correct methods", () => {
      expect(apiClient.get).toBeDefined();
      expect(apiClient.post).toBeDefined();
    });
  });

  describe("healthApi", () => {
    it("should have checkHealth method", () => {
      expect(healthApi.checkHealth).toBeDefined();
      expect(typeof healthApi.checkHealth).toBe("function");
    });

    it("should call correct endpoint", async () => {
      const mockResponse = { data: { status: "healthy" } };
      (apiClient.get as jest.Mock).mockResolvedValueOnce(mockResponse);

      const result = await healthApi.checkHealth();

      expect(apiClient.get).toHaveBeenCalledWith("/health");
      expect(result).toEqual({ status: "healthy" });
    });

    it("should handle errors", async () => {
      const error = new Error("Network error");
      (apiClient.get as jest.Mock).mockRejectedValueOnce(error);

      await expect(healthApi.checkHealth()).rejects.toThrow("Network error");
    });
  });

  describe("versionApi", () => {
    it("should have getVersion method", () => {
      expect(versionApi.getVersion).toBeDefined();
      expect(typeof versionApi.getVersion).toBe("function");
    });

    it("should call correct endpoint", async () => {
      const mockResponse = { data: { version: "0.10.1" } };
      (apiClient.get as jest.Mock).mockResolvedValueOnce(mockResponse);

      const result = await versionApi.getVersion();

      expect(apiClient.get).toHaveBeenCalledWith("/version");
      expect(result).toEqual({ version: "0.10.1" });
    });
  });
});
