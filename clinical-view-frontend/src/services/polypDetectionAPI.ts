/**
 * API Service for Polyp Detection
 * Connects the frontend to the YOLO API backend
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const USE_MOCK = import.meta.env.VITE_MOCK_API === 'true';

export interface Detection {
  id: number;
  class: string;
  confidence: number;
  bbox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: {
    x: number;
    y: number;
  };
  area: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface DetectionResponse {
  detections: Detection[];
  annotatedImage: string;
  processingTime: number;
  imageSize: {
    width: number;
    height: number;
  };
  totalDetections: number;
  confidenceStats: {
    high: number;
    medium: number;
    low: number;
  };
}

export interface HealthCheckResponse {
  status: string;
  model_loaded: boolean;
  model_path: string;
}

class PolypDetectionAPI {
  private baseURL: string;
  private useMock: boolean;

  constructor() {
    this.baseURL = API_BASE_URL;
    this.useMock = USE_MOCK;
  }

  /**
   * Check API health status
   */
  async checkHealth(): Promise<HealthCheckResponse> {
    if (this.useMock) {
      return {
        status: 'healthy',
        model_loaded: true,
        model_path: 'mock_model',
      };
    }

    try {
      const response = await fetch(`${this.baseURL}/api/health`);
      if (!response.ok) {
        throw new Error('Health check failed');
      }
      return await response.json();
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  /**
   * Upload image and detect polyps
   */
  async detectPolyps(imageFile: File): Promise<DetectionResponse> {
    if (this.useMock) {
      // Return mock data for testing
      return this.getMockDetectionResponse();
    }

    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      const response = await fetch(`${this.baseURL}/api/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Detection failed');
      }

      const data: DetectionResponse = await response.json();
      return data;
    } catch (error) {
      console.error('Detection error:', error);
      throw error;
    }
  }

  /**
   * Mock detection response for testing
   */
  private getMockDetectionResponse(): DetectionResponse {
    return {
      detections: [
        {
          id: 1,
          class: 'Adenomatous Polyp',
          confidence: 0.982,
          bbox: { x: 150, y: 120, width: 80, height: 60 },
          center: { x: 190, y: 150 },
          area: 4800,
          riskLevel: 'high',
        },
        {
          id: 2,
          class: 'Hyperplastic Polyp',
          confidence: 0.745,
          bbox: { x: 300, y: 200, width: 50, height: 40 },
          center: { x: 325, y: 220 },
          area: 2000,
          riskLevel: 'low',
        },
      ],
      annotatedImage: 'data:image/jpeg;base64,mock_base64_image',
      processingTime: 0.4,
      imageSize: { width: 640, height: 480 },
      totalDetections: 2,
      confidenceStats: { high: 1, medium: 1, low: 0 },
    };
  }
}

// Export singleton instance
export const polypAPI = new PolypDetectionAPI();

// Export helper functions
export const isAPIHealthy = async (): Promise<boolean> => {
  try {
    const health = await polypAPI.checkHealth();
    return health.status === 'healthy' && health.model_loaded;
  } catch {
    return false;
  }
};

export const analyzeImage = async (file: File): Promise<DetectionResponse> => {
  return polypAPI.detectPolyps(file);
};
