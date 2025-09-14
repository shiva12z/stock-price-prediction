// API Configuration and utilities
const API_BASE_URL = import.meta.env.PROD ? '' : 'http://localhost:5000';

class APIClient {
  constructor() {
    this.baseURL = API_BASE_URL;
    this.cache = new Map();
    this.cacheTimeout = 60000; // 1 minute cache
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const cacheKey = `${endpoint}${JSON.stringify(options.params || {})}`;
    
    // Check cache first
    if (this.cache.has(cacheKey)) {
      const { data, timestamp } = this.cache.get(cacheKey);
      if (Date.now() - timestamp < this.cacheTimeout) {
        return data;
      }
      this.cache.delete(cacheKey);
    }

    try {
      const config = {
        method: options.method || 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      };

      if (options.params) {
        const searchParams = new URLSearchParams(options.params);
        const separator = url.includes('?') ? '&' : '?';
        config.url = `${url}${separator}${searchParams}`;
      }

      const response = await fetch(config.url || url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      // Cache successful responses
      this.cache.set(cacheKey, { data, timestamp: Date.now() });
      
      return data;
    } catch (error) {
      console.error(`API request failed for ${endpoint}:`, error);
      throw error;
    }
  }

  async searchStocks(query) {
    return this.request('/search', { params: { q: query } });
  }

  async getStock(symbol, lite = false) {
    return this.request(`/stock/${symbol}`, { params: { lite: lite ? '1' : '0' } });
  }

  async getChartData(symbol) {
    return this.request(`/chart-data/${symbol}`);
  }

  async predictPrice(symbol, model = 'lr', fromDate = null, toDate = null) {
    const params = { symbol, model };
    if (fromDate) params.from = fromDate;
    if (toDate) params.to = toDate;
    return this.request('/predict', { params });
  }
}

export const apiClient = new APIClient();