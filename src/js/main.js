// Main application entry point
import { apiClient } from './api.js';
import { chartManager } from './charts.js';
import { SearchDropdown, Toast, LoadingSpinner } from './components.js';

class StockWiseApp {
  constructor() {
    this.currentStock = null;
    this.searchDropdown = null;
    this.init();
  }

  async init() {
    try {
      this.setupEventListeners();
      this.setupSearch();
      await this.loadInitialData();
      this.setupGoogleSignIn();
    } catch (error) {
      console.error('Failed to initialize app:', error);
      Toast.error('Failed to initialize application');
    }
  }

  setupEventListeners() {
    // Chart period buttons
    document.querySelectorAll('[data-period]').forEach(button => {
      button.addEventListener('click', (e) => {
        // Update active state
        document.querySelectorAll('[data-period]').forEach(btn => 
          btn.classList.remove('active'));
        e.target.classList.add('active');
        
        // Update chart
        this.updateMarketChart(e.target.dataset.period);
      });
    });

    // Quick action buttons
    window.showStockAnalysis = () => this.showStockAnalysis();
    window.showPredictions = () => this.showPredictions();
    window.showPortfolio = () => this.showPortfolio();
  }

  setupSearch() {
    const searchInput = document.getElementById('stockSearch');
    if (searchInput) {
      this.searchDropdown = new SearchDropdown(searchInput, (result) => {
        this.selectStock(result);
      });
    }
  }

  setupGoogleSignIn() {
    if (typeof google !== 'undefined' && google.accounts) {
      try {
        google.accounts.id.initialize({
          client_id: "366312949881-c1pvjbgeqt614i2vk9ekdfd0n7mc6v8m.apps.googleusercontent.com",
          callback: this.handleGoogleSignIn.bind(this),
          use_fedcm_for_prompt: false,
          auto_select: false
        });
      } catch (error) {
        console.error('Google Sign-In initialization failed:', error);
      }
    }
  }

  handleGoogleSignIn(response) {
    try {
      if (response && response.credential) {
        localStorage.setItem('google_id_token', response.credential);
        Toast.success('Successfully signed in with Google');
        // Update UI to show signed-in state
        this.updateUserInterface();
      }
    } catch (error) {
      console.error('Google Sign-In error:', error);
      Toast.error('Google Sign-In failed');
    }
  }

  async loadInitialData() {
    try {
      // Load market overview chart with sample data
      await this.updateMarketChart('1d');
      
      // Load top performers
      await this.loadTopPerformers();
      
      // Load recent predictions
      await this.loadRecentPredictions();
      
    } catch (error) {
      console.error('Failed to load initial data:', error);
    }
  }

  async updateMarketChart(period = '1d') {
    const chartCanvas = document.getElementById('marketChart');
    if (!chartCanvas) return;

    try {
      // For demo purposes, generate sample data
      // In production, this would fetch real market data
      const sampleData = this.generateSampleMarketData(period);
      
      chartManager.createPriceChart('marketChart', sampleData);
    } catch (error) {
      console.error('Failed to update market chart:', error);
      Toast.error('Failed to load market data');
    }
  }

  generateSampleMarketData(period) {
    const periods = {
      '1d': { points: 24, label: 'Hour' },
      '1w': { points: 7, label: 'Day' },
      '1m': { points: 30, label: 'Day' },
      '3m': { points: 90, label: 'Day' },
      '1y': { points: 365, label: 'Day' }
    };

    const config = periods[period] || periods['1d'];
    const labels = [];
    const prices = [];
    
    let basePrice = 15000;
    const now = new Date();

    for (let i = config.points - 1; i >= 0; i--) {
      const date = new Date(now);
      if (period === '1d') {
        date.setHours(date.getHours() - i);
        labels.push(date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }));
      } else {
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
      }
      
      // Generate realistic price movement
      const change = (Math.random() - 0.5) * 200;
      basePrice += change;
      prices.push(Math.max(basePrice, 10000)); // Ensure positive prices
    }

    return { labels, prices };
  }

  async loadTopPerformers() {
    const container = document.getElementById('topPerformers');
    if (!container) return;

    try {
      // Sample top performers data
      const performers = [
        { symbol: 'AAPL', name: 'Apple Inc.', change: 5.67, price: 182.52 },
        { symbol: 'MSFT', name: 'Microsoft Corp.', change: 3.21, price: 378.85 },
        { symbol: 'GOOGL', name: 'Alphabet Inc.', change: 2.89, price: 142.56 },
        { symbol: 'TSLA', name: 'Tesla Inc.', change: 8.45, price: 248.42 },
        { symbol: 'NVDA', name: 'NVIDIA Corp.', change: 4.12, price: 875.28 }
      ];

      container.innerHTML = performers.map(stock => `
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0; border-bottom: 1px solid var(--color-border);">
          <div>
            <div style="font-weight: 600; color: var(--color-text-primary);">${stock.symbol}</div>
            <div style="font-size: 0.875rem; color: var(--color-text-secondary);">${stock.name}</div>
          </div>
          <div style="text-align: right;">
            <div style="font-weight: 600; color: var(--color-text-primary);">$${stock.price.toFixed(2)}</div>
            <div style="font-size: 0.875rem; color: var(--color-success);">+${stock.change}%</div>
          </div>
        </div>
      `).join('');
    } catch (error) {
      console.error('Failed to load top performers:', error);
      container.innerHTML = '<p style="color: var(--color-text-secondary); text-align: center; padding: 1rem;">Failed to load data</p>';
    }
  }

  async loadRecentPredictions() {
    const container = document.getElementById('recentPredictions');
    if (!container) return;

    try {
      // Sample predictions data
      const predictions = [
        { symbol: 'AAPL', model: 'Random Forest', accuracy: 87.5, date: '2024-01-15' },
        { symbol: 'MSFT', model: 'Linear Regression', accuracy: 82.3, date: '2024-01-14' },
        { symbol: 'GOOGL', model: 'ARIMA', accuracy: 79.8, date: '2024-01-14' },
        { symbol: 'TSLA', model: 'Random Forest', accuracy: 91.2, date: '2024-01-13' },
        { symbol: 'NVDA', model: 'Linear Regression', accuracy: 85.7, date: '2024-01-13' }
      ];

      container.innerHTML = predictions.map(pred => `
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 0; border-bottom: 1px solid var(--color-border);">
          <div>
            <div style="font-weight: 600; color: var(--color-text-primary);">${pred.symbol}</div>
            <div style="font-size: 0.875rem; color: var(--color-text-secondary);">${pred.model}</div>
          </div>
          <div style="text-align: right;">
            <div style="font-weight: 600; color: var(--color-primary);">${pred.accuracy}%</div>
            <div style="font-size: 0.875rem; color: var(--color-text-secondary);">${new Date(pred.date).toLocaleDateString()}</div>
          </div>
        </div>
      `).join('');
    } catch (error) {
      console.error('Failed to load recent predictions:', error);
      container.innerHTML = '<p style="color: var(--color-text-secondary); text-align: center; padding: 1rem;">Failed to load data</p>';
    }
  }

  async selectStock(stock) {
    this.currentStock = stock;
    Toast.success(`Selected ${stock.symbol} - ${stock.name}`);
    
    // You could navigate to a detailed view or update the current page
    // For now, we'll just show a success message
  }

  showStockAnalysis() {
    if (this.currentStock) {
      window.location.href = `./stocks.html?symbol=${this.currentStock.symbol}`;
    } else {
      Toast.warning('Please search and select a stock first');
      document.getElementById('stockSearch').focus();
    }
  }

  showPredictions() {
    window.location.href = './predictions.html';
  }

  showPortfolio() {
    window.location.href = './portfolio.html';
  }

  updateUserInterface() {
    // Update UI elements based on authentication state
    const isSignedIn = localStorage.getItem('google_id_token');
    
    if (isSignedIn) {
      // Update user avatar, show personalized content, etc.
      console.log('User is signed in, updating interface...');
    }
  }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new StockWiseApp();
});

// Handle page visibility changes for performance optimization
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    // Pause expensive operations when page is hidden
    console.log('Page hidden, pausing operations...');
  } else {
    // Resume operations when page becomes visible
    console.log('Page visible, resuming operations...');
  }
});