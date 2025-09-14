// Chart utilities and configurations
import Chart from 'chart.js/auto';
import 'chartjs-adapter-date-fns';

// Chart.js default configuration
Chart.defaults.font.family = 'Manrope, "Noto Sans", sans-serif';
Chart.defaults.color = '#a2c398';
Chart.defaults.backgroundColor = 'rgba(84, 210, 45, 0.1)';

export const chartTheme = {
  colors: {
    primary: '#54d22d',
    secondary: '#ff6b6b',
    accent: '#ffc107',
    background: '#1e2a1b',
    text: '#a2c398',
    grid: '#426039',
    tooltip: 'rgba(30, 42, 27, 0.95)'
  }
};

export const defaultChartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  interaction: {
    mode: 'index',
    intersect: false,
  },
  plugins: {
    legend: {
      labels: {
        color: chartTheme.colors.text,
        usePointStyle: true,
        padding: 20
      }
    },
    tooltip: {
      backgroundColor: chartTheme.colors.tooltip,
      titleColor: '#ffffff',
      bodyColor: chartTheme.colors.text,
      borderColor: chartTheme.colors.grid,
      borderWidth: 1,
      cornerRadius: 8,
      displayColors: true,
      callbacks: {
        label: function(context) {
          const label = context.dataset.label || '';
          const value = context.parsed.y;
          return `${label}: $${value.toFixed(2)}`;
        }
      }
    }
  },
  scales: {
    x: {
      ticks: {
        color: chartTheme.colors.text,
        maxTicksLimit: 10
      },
      grid: {
        color: chartTheme.colors.grid,
        drawBorder: false
      }
    },
    y: {
      ticks: {
        color: chartTheme.colors.text,
        callback: function(value) {
          return '$' + value.toFixed(2);
        }
      },
      grid: {
        color: chartTheme.colors.grid,
        drawBorder: false
      }
    }
  }
};

export class ChartManager {
  constructor() {
    this.charts = new Map();
  }

  createPriceChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    // Destroy existing chart if it exists
    if (this.charts.has(canvasId)) {
      this.charts.get(canvasId).destroy();
    }

    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [{
          label: 'Price',
          data: data.prices,
          borderColor: chartTheme.colors.primary,
          backgroundColor: 'rgba(84, 210, 45, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 6
        }]
      },
      options: {
        ...defaultChartOptions,
        plugins: {
          ...defaultChartOptions.plugins,
          title: {
            display: true,
            text: 'Stock Price History',
            color: '#ffffff',
            font: {
              size: 16,
              weight: 'bold'
            }
          }
        }
      }
    });

    this.charts.set(canvasId, chart);
    return chart;
  }

  createVolumeChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    if (this.charts.has(canvasId)) {
      this.charts.get(canvasId).destroy();
    }

    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: data.labels,
        datasets: [{
          label: 'Volume',
          data: data.volumes,
          backgroundColor: 'rgba(84, 210, 45, 0.6)',
          borderColor: chartTheme.colors.primary,
          borderWidth: 1
        }]
      },
      options: {
        ...defaultChartOptions,
        plugins: {
          ...defaultChartOptions.plugins,
          title: {
            display: true,
            text: 'Trading Volume',
            color: '#ffffff',
            font: {
              size: 16,
              weight: 'bold'
            }
          },
          tooltip: {
            ...defaultChartOptions.plugins.tooltip,
            callbacks: {
              label: function(context) {
                const value = context.parsed.y;
                return `Volume: ${value.toLocaleString()}`;
              }
            }
          }
        },
        scales: {
          ...defaultChartOptions.scales,
          y: {
            ...defaultChartOptions.scales.y,
            ticks: {
              ...defaultChartOptions.scales.y.ticks,
              callback: function(value) {
                return value.toLocaleString();
              }
            }
          }
        }
      }
    });

    this.charts.set(canvasId, chart);
    return chart;
  }

  createPredictionChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    if (this.charts.has(canvasId)) {
      this.charts.get(canvasId).destroy();
    }

    const datasets = [];
    
    // Historical data
    if (data.actual_prices && data.actual_prices.length > 0) {
      datasets.push({
        label: 'Historical Prices',
        data: data.actual_prices,
        borderColor: chartTheme.colors.primary,
        backgroundColor: 'rgba(84, 210, 45, 0.1)',
        borderWidth: 2,
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4
      });
    }

    // Predicted data
    if (data.predicted_prices && data.predicted_prices.length > 0) {
      datasets.push({
        label: 'Predicted Prices',
        data: data.predicted_prices,
        borderColor: chartTheme.colors.secondary,
        backgroundColor: 'rgba(255, 107, 107, 0.1)',
        borderWidth: 2,
        borderDash: [5, 5],
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        pointHoverRadius: 4
      });
    }

    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: datasets
      },
      options: {
        ...defaultChartOptions,
        plugins: {
          ...defaultChartOptions.plugins,
          title: {
            display: true,
            text: 'Price Prediction',
            color: '#ffffff',
            font: {
              size: 16,
              weight: 'bold'
            }
          }
        }
      }
    });

    this.charts.set(canvasId, chart);
    return chart;
  }

  destroyChart(canvasId) {
    if (this.charts.has(canvasId)) {
      this.charts.get(canvasId).destroy();
      this.charts.delete(canvasId);
    }
  }

  destroyAllCharts() {
    this.charts.forEach(chart => chart.destroy());
    this.charts.clear();
  }
}

export const chartManager = new ChartManager();