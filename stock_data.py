import requests
import pandas as pd
import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dateutil.parser import parse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Finnhub API key from environment variables
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
if not FINNHUB_API_KEY:
    print("Warning: FINNHUB_API_KEY not found in environment variables. Some features may not work.")

def get_stock_data(
    ticker_symbol: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    period: str = "1y"
) -> Optional[Dict]:
    """
    Fetch stock data for the given ticker symbol and date range.
    
    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        start_date (str/datetime, optional): Start date in 'YYYY-MM-DD' format or datetime object
        end_date (str/datetime, optional): End date in 'YYYY-MM-DD' format or datetime object
        period (str): Time period to fetch data for if no dates provided
                     Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
                     
    Returns:
        dict: Dictionary containing stock data and metadata or None if error occurs
    """
    try:
        # Resolve period or date range to Yahoo chart API parameters
        range_map = {
            '1d': ('1d', '1m'),
            '5d': ('5d', '5m'),
            '1mo': ('1mo', '1d'),
            '3mo': ('3mo', '1d'),
            '6mo': ('6mo', '1d'),
            '1y': ('1y', '1d'),
            '2y': ('2y', '1d'),
            '5y': ('5y', '1wk'),
            '10y': ('10y', '1mo'),
            'ytd': ('ytd', '1d'),
            'max': ('max', '1mo'),
        }

        if isinstance(start_date, str):
            start_date = parse(start_date).date()
        if isinstance(end_date, str):
            end_date = parse(end_date).date()

        params = {}
        if start_date and end_date:
            from datetime import datetime as dt
            import time as _t
            start_ts = int(_t.mktime(dt(start_date.year, start_date.month, start_date.day).timetuple()))
            end_ts = int(_t.mktime(dt(end_date.year, end_date.month, end_date.day).timetuple()))
            params = {'period1': start_ts, 'period2': end_ts, 'interval': '1d'}
        else:
            r, i = range_map.get(period, ('1y', '1d'))
            params = {'range': r, 'interval': i}

        url = f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker_symbol}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'https://finance.yahoo.com/'
        }
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        result = (data.get('chart') or {}).get('result') or []
        if not result:
            print(f"No data found for {ticker_symbol}.")
            return None
        res = result[0]
        ts = res.get('timestamp') or []
        quotes = ((res.get('indicators') or {}).get('quote') or [{}])[0]
        opens = quotes.get('open') or []
        highs = quotes.get('high') or []
        lows = quotes.get('low') or []
        closes = quotes.get('close') or []
        volumes = quotes.get('volume') or []

        # Build pandas DataFrame similar to yfinance output
        import pandas as _pd
        rows = []
        for i, t in enumerate(ts):
            try:
                dt = datetime.utcfromtimestamp(int(t))
            except Exception:
                continue
            rows.append({
                'Date': dt,
                'Open': opens[i] if i < len(opens) else None,
                'High': highs[i] if i < len(highs) else None,
                'Low': lows[i] if i < len(lows) else None,
                'Close': closes[i] if i < len(closes) else None,
                'Volume': volumes[i] if i < len(volumes) else None,
            })
        if not rows:
            print(f"No rows built for {ticker_symbol}.")
            return None
        df = _pd.DataFrame(rows)
        df = df.set_index('Date')
        df = df.dropna(how='all')
        if df.empty:
            print(f"No data after cleaning for {ticker_symbol}.")
            return None

        latest = df.iloc[-1]
        first = df.iloc[0]

        return {
            'ticker': ticker_symbol,
            'history': df,
            'metrics': {
                'start_date': df.index[0].strftime('%Y-%m-%d'),
                'end_date': df.index[-1].strftime('%Y-%m-%d'),
                'days_of_data': len(df),
                'price_change': latest['Close'] - first['Close'],
                'percent_change': ((latest['Close'] - first['Close']) / first['Close']) * 100,
                'average_volume': df['Volume'].mean(),
            },
            'info': {
                'company_name': ticker_symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'current_price': latest['Close'],
                'market_cap': 'N/A',
                'dividend_yield': 0,
                'fifty_two_week': {
                    'high': 'N/A',
                    'low': 'N/A'
                }
            }
        }
    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {str(e)}")
        return None

def get_multiple_stocks(
    tickers: List[str],
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    period: str = "1y"
) -> Dict[str, Dict]:
    """
    Fetch data for multiple stock tickers with the same date range.
    
    Args:
        tickers (List[str]): List of ticker symbols
        start_date (str/datetime, optional): Start date in 'YYYY-MM-DD' format or datetime object
        end_date (str/datetime, optional): End date in 'YYYY-MM-DD' format or datetime object
        period (str): Time period to fetch data for if no dates provided
                     
    Returns:
        dict: Dictionary with ticker symbols as keys and their data as values
    """
    results = {}
    for ticker in tickers:
        data = get_stock_data(ticker, start_date, end_date, period)
        if data:
            results[ticker] = data
    return results

def get_stock_data_in_chunks(
    tickers: List[str],
    start_date: Union[str, datetime],
    end_date: Optional[Union[str, datetime]] = None,
    chunk_size: int = 5,
    delay: int = 1
) -> Dict[str, Dict]:
    """
    Fetch data for multiple tickers in chunks to avoid rate limiting.
    
    Args:
        tickers (List[str]): List of ticker symbols
        start_date (str/datetime): Start date in 'YYYY-MM-DD' format or datetime object
        end_date (str/datetime, optional): End date in 'YYYY-MM-DD' format or datetime object
        chunk_size (int): Number of tickers to fetch in each chunk
        delay (int): Delay in seconds between chunks
        
    Returns:
        dict: Dictionary with ticker symbols as keys and their data as values
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = {}
    
    # Process tickers in chunks
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        
        # Use ThreadPoolExecutor to fetch data in parallel
        with ThreadPoolExecutor(max_workers=chunk_size) as executor:
            future_to_ticker = {
                executor.submit(get_stock_data, ticker, start_date, end_date): ticker 
                for ticker in chunk
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    data = future.result()
                    if data:
                        results[ticker] = data
                except Exception as e:
                    print(f"Error processing {ticker}: {str(e)}")
        
        # Add delay between chunks to avoid rate limiting
        if i + chunk_size < len(tickers):
            time.sleep(delay)
    
    return results

def format_stock_data_for_display(data: Dict) -> str:
    """Format stock data for display in console."""
    if not data:
        return "No data available"
        
    output = [
        f"\n=== {data['ticker']} ===",
        f"Company: {data['info']['company_name']}",
        f"Sector: {data['info']['sector']}",
        f"Industry: {data['info']['industry']}",
        f"Current Price: ${data['info']['current_price']:,.2f}",
        f"Market Cap: ${data['info']['market_cap']:,}" if isinstance(data['info']['market_cap'], (int, float)) else "Market Cap: N/A",
        f"52-Week Range: ${data['info']['fifty_two_week']['low']} - ${data['info']['fifty_two_week']['high']}",
        f"Dividend Yield: {data['info']['dividend_yield']:.2f}%",
        f"\nDate Range: {data['metrics']['start_date']} to {data['metrics']['end_date']}",
        f"Price Change: {data['metrics']['price_change']:+,.2f} ({data['metrics']['percent_change']:+.2f}%)",
        f"Average Daily Volume: {data['metrics']['average_volume']:,.0f}",
        "\nLast 5 days of data:",
        str(data['history'].tail())
    ]
    return "\n".join(output)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch stock market data')
    parser.add_argument('tickers', nargs='+', help='Stock ticker symbols (e.g., AAPL MSFT GOOGL)')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD, defaults to today)')
    parser.add_argument('--period', default='1y', 
                       help='Time period if no dates provided (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)')
    
    args = parser.parse_args()
    
    print(f"Fetching data for {', '.join(args.tickers)}...")
    
    if args.start or args.end:
        # Use date range if either start or end date is provided
        stocks_data = get_multiple_stocks(
            args.tickers, 
            start_date=args.start, 
            end_date=args.end
        )
    else:
        # Use period if no dates provided
        stocks_data = get_multiple_stocks(args.tickers, period=args.period)
    
    for ticker, data in stocks_data.items():
        if data:
            print(format_stock_data_for_display(data))
            print("\n" + "="*50 + "\n")
