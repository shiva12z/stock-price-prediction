# StockWise - Stock Prediction with Charts

A comprehensive stock analysis and prediction platform that combines machine learning models with interactive chart visualizations.

## Features

### 📊 Interactive Charts
- **Historical Price Charts**: Visualize stock performance over time
- **Prediction Charts**: Compare historical vs. predicted prices
- **Volume Charts**: Analyze trading volume patterns
- **Model Comparison Charts**: Compare performance of different ML models

### 🤖 Machine Learning Models
- **Linear Regression**: Simple trend-based predictions
- **Random Forest**: Ensemble learning for robust predictions
- **ARIMA**: Time series analysis for trend forecasting

### 🎨 Chart Features
- **Responsive Design**: Charts adapt to different screen sizes
- **Interactive Tooltips**: Hover over data points for detailed information
- **Chart Type Toggle**: Switch between line and candlestick views
- **Zoom & Pan**: Interactive chart navigation
- **Dark Theme**: Consistent with the app's design

## Getting Started

### Prerequisites
- Python 3.7+
- Flask
- Required packages (see requirements.txt)

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Flask server: `python app.py`

### Usage

#### 1. Search for Stocks
- Use the search bar to find stocks by symbol or company name
- Click on a stock to view detailed information

#### 2. View Charts
- **Overview Tab**: See stock performance and volume charts
- **Prediction Tab**: Access prediction tools and charts

#### 3. Run Predictions
- Select date range and model type
- Click "Predict" for single model prediction
- Click "Compare All Models" to see model performance comparison

#### 4. Chart Interactions
- **Hover**: View detailed data points
- **Click & Drag**: Pan across the chart
- **Scroll**: Zoom in/out
- **Toggle**: Switch between chart types

## Chart Types

### Line Charts
- Smooth curves with fill areas
- Perfect for trend visualization
- Interactive tooltips on hover

### Bar Charts
- Used for volume and model comparison
- Clear data representation
- Color-coded for easy identification

### Multi-Dataset Charts
- Compare multiple data series
- Historical vs. predicted prices
- Different line styles for clarity

## API Endpoints

### `/chart-data/<symbol>`
Returns historical price and volume data for charting.

### `/predict`
Runs ML model predictions with chart data included.

## Technical Details

### Frontend
- **Chart.js**: Interactive chart library
- **Responsive Design**: Mobile-friendly interface
- **Dark Theme**: Consistent color scheme

### Backend
- **Flask**: Python web framework
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning models
- **Statsmodels**: ARIMA implementation

### Data Sources
- **Yahoo Finance API**: Real-time stock data
- **Historical Data**: 6-month price history
- **Volume Data**: Trading volume analysis

## Chart Customization

### Colors
- Primary: `#54d22d` (Green)
- Secondary: `#ff6b6b` (Red)
- Accent: `#ffc107` (Yellow)
- Background: `#1e2a1b` (Dark Green)

### Styling
- Grid lines: `#426039`
- Text: `#a2c398`
- Borders: `#426039`

## Troubleshooting

### Charts Not Loading
1. Check browser console for errors
2. Verify Chart.js is loaded
3. Ensure canvas elements exist

### Data Not Displaying
1. Check API endpoints
2. Verify stock symbol is valid
3. Check network connectivity

### Performance Issues
1. Reduce chart data points
2. Use appropriate chart types
3. Optimize data fetching

## Future Enhancements

- [ ] Real-time chart updates
- [ ] More chart types (candlestick, OHLC)
- [ ] Technical indicators
- [ ] Export chart data
- [ ] Custom time ranges
- [ ] Multiple stock comparison

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

---

**Note**: This is an educational tool. Do not use predictions for actual trading decisions.

