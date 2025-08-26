from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import sys
from datetime import datetime, timedelta
import os
import pandas as pd
import requests
import time
from typing import Any, Callable, Optional, Tuple

try:
    # urllib3 Retry is useful for robust HTTP retries on 429/5xx
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception:
    HTTPAdapter = None
    Retry = None

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS
CORS(app)

# Create a shared requests session with desktop User-Agent to reduce Yahoo blocks
def get_http_session():
    try:
        sess = requests.Session()
        sess.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Referer': 'https://finance.yahoo.com/'
        })
        # Attach retry adapter when available
        if HTTPAdapter and Retry:
            retry = Retry(
                total=3,
                backoff_factor=0.8,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["GET", "POST"],
                raise_on_status=False,
            )
            adapter = HTTPAdapter(max_retries=retry)
            sess.mount('http://', adapter)
            sess.mount('https://', adapter)
        return sess
    except Exception:
        return None

# Simple in-memory cache to avoid hammering Yahoo endpoints for the same symbol
_symbol_cache: dict[str, Tuple[float, dict]] = {}
_CACHE_TTL_SECONDS = 120

def _cache_get(symbol: str) -> Optional[dict]:
    try:
        entry = _symbol_cache.get(symbol)
        if not entry:
            return None
        ts, data = entry
        if (time.time() - ts) <= _CACHE_TTL_SECONDS:
            return data
        # expired
        _symbol_cache.pop(symbol, None)
        return None
    except Exception:
        return None

def _cache_set(symbol: str, data: dict) -> None:
    try:
        _symbol_cache[symbol] = (time.time(), data)
    except Exception:
        pass

def _retry(fn: Callable[[], Any], attempts: int = 3, first_sleep: float = 0.5, factor: float = 2.0) -> Any:
    last_err = None
    sleep_s = first_sleep
    for _ in range(max(1, attempts)):
        try:
            return fn()
        except Exception as e:
            last_err = e
            try:
                time.sleep(sleep_s)
            except Exception:
                pass
            sleep_s *= factor
    if last_err:
        raise last_err
    return None

@app.route('/search')
def search_stocks():
    """Search stocks by name/symbol across multiple regions.

    Uses Yahoo Finance's public search API for better name lookup and
    supports Indian exchanges (NSE/BSE) out of the box.
    """
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])

    import requests

    def yahoo_search(q: str, region: str):
        try:
            url = 'https://query1.finance.yahoo.com/v1/finance/search'
            params = {
                'q': q,
                'quotesCount': 20,
                'newsCount': 0,
                'listsCount': 0,
                'enableFuzzyQuery': 'false',
                'quotesQueryId': 'tss_match_phrase_query',
                'multiQuoteQueryId': 'multi_quote_single_token_query',
                'lang': 'en-IN' if region == 'IN' else 'en-US',
                'region': region,
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Referer': 'https://finance.yahoo.com/'
            }
            r = requests.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json() or {}
            results = []
            for q in data.get('quotes', [])[:10]:
                symbol = q.get('symbol') or ''
                if not symbol:
                    continue
                results.append({
                    'symbol': symbol,
                    'name': q.get('shortname') or q.get('longname') or q.get('name') or symbol,
                    'exchange': q.get('exchDisp') or q.get('exchange') or 'Unknown Exchange',
                })
            return results
        except Exception as e:
            logger.warning(f"Yahoo search failed for region {region}: {e}")
            return []

    try:
        # Search across regions and merge unique symbols while preserving order
        regions = ['US', 'IN', 'GB']
        combined = []
        seen = set()
        for reg in regions:
            for item in yahoo_search(query, reg):
                if item['symbol'] in seen:
                    continue
                combined.append(item)
                seen.add(item['symbol'])

        return jsonify(combined[:10])
    except Exception as e:
        logger.error(f"Error searching for {query}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/stock/<symbol>')
def get_stock(symbol):
    try:
        lite = (request.args.get('lite', '0') in ['1', 'true', 'True'])
        # Serve from cache when available
        cached = _cache_get(symbol)
        if cached:
            return jsonify(cached)

        session = get_http_session()
        
        # No yfinance: use Yahoo chart API via requests
        info = {}
        fast = {}

        # Determine currency and human exchange from common Yahoo suffixes
        currency_code = 'USD'
        derived_exchange = None
        try:
            suffix_map = {
                '.NS': ('INR', 'NSE'),
                '.BO': ('INR', 'BSE'),
                '.L': ('GBP', 'LSE'),
                '.TO': ('CAD', 'TSX'),
                '.HK': ('HKD', 'HKEX'),
                '.AX': ('AUD', 'ASX'),
                '.SZ': ('CNY', 'SZSE'),
                '.SS': ('CNY', 'SSE'),
                '.T': ('JPY', 'TSE'),
                '.PA': ('EUR', 'EURONEXT PARIS'),
                '.F': ('EUR', 'FRANKFURT'),
                '.DE': ('EUR', 'XETRA'),
            }
            for suf, (cur, exch) in suffix_map.items():
                if symbol.upper().endswith(suf):
                    currency_code = cur
                    derived_exchange = exch
                    break
        except Exception:
            pass

        # Helper: fetch Yahoo v7 quote for market cap and basic fields
        def fetch_quote(sym: str):
            try:
                url = 'https://query1.finance.yahoo.com/v7/finance/quote'
                params = {'symbols': sym}
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                    'Accept': 'application/json, text/plain, */*',
                    'Referer': 'https://finance.yahoo.com/'
                }
                s = session or requests
                r = s.get(url, params=params, headers=headers, timeout=10)
                r.raise_for_status()
                data = r.json() or {}
                res = ((data.get('quoteResponse') or {}).get('result') or [])
                return res[0] if res else {}
            except Exception:
                return {}

        quote = fetch_quote(symbol)
        try:
            if quote.get('currency'):
                currency_code = str(quote.get('currency')).upper()
        except Exception:
            pass
        try:
            if quote.get('fullExchangeName') and not derived_exchange:
                derived_exchange = quote.get('fullExchangeName')
        except Exception:
            pass

        # Get historical data for the last 30 days (best-effort) via Yahoo chart API
        history = []
        def fetch_chart(range_str: str, interval: str):
            url = f'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
            params = {'range': range_str, 'interval': interval}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Referer': 'https://finance.yahoo.com/'
            }
            s = session or requests
            r = s.get(url, params=params, headers=headers, timeout=10)
            r.raise_for_status()
            return r.json()

        try:
            if not lite:
                data_1mo = _retry(lambda: fetch_chart('1mo', '1d'), attempts=2)
                result = (data_1mo or {}).get('chart', {}).get('result', [])
                if result:
                    r0 = result[0]
                    # Prefer currency from Yahoo meta if present
                    try:
                        meta_cur = ((r0 or {}).get('meta') or {}).get('currency')
                        if meta_cur:
                            currency_code = str(meta_cur).upper()
                    except Exception:
                        pass
                    ts = r0.get('timestamp') or []
                    quotes = ((r0.get('indicators') or {}).get('quote') or [{}])[0]
                    opens = quotes.get('open') or []
                    highs = quotes.get('high') or []
                    lows = quotes.get('low') or []
                    closes = quotes.get('close') or []
                    volumes = quotes.get('volume') or []
                    for i, t in enumerate(ts):
                        try:
                            date = datetime.utcfromtimestamp(int(t)).strftime('%Y-%m-%d')
                        except Exception:
                            continue
                        history.append({
                            'date': date,
                            'open': round(float(opens[i]), 2) if i < len(opens) and opens[i] is not None else None,
                            'high': round(float(highs[i]), 2) if i < len(highs) and highs[i] is not None else None,
                            'low': round(float(lows[i]), 2) if i < len(lows) and lows[i] is not None else None,
                            'close': round(float(closes[i]), 2) if i < len(closes) and closes[i] is not None else None,
                            'volume': int(volumes[i]) if i < len(volumes) and volumes[i] is not None else None,
                        })
        except Exception:
            history = []

        # Derive robust fields
        def pick(*keys, srcs):
            for key in keys:
                for src in srcs:
                    if key in src and src.get(key) is not None:
                        return src.get(key)
            return 0

        last_price = pick('regularMarketPrice', 'currentPrice', 'last_price', srcs=[quote, fast, info])
        open_price = pick('regularMarketOpen', 'open', srcs=[quote, fast, info])
        day_high = pick('regularMarketDayHigh', 'day_high', srcs=[quote, fast, info])
        day_low = pick('regularMarketDayLow', 'day_low', srcs=[quote, fast, info])
        prev_close = pick('regularMarketPreviousClose', 'previous_close', srcs=[quote, fast, info])
        market_cap = pick('marketCap', 'market_cap', srcs=[quote, fast, info])
        volume = pick('regularMarketVolume', 'last_volume', 'volume', srcs=[quote, fast, info])
        change = 0
        change_percent = 0
        try:
            if last_price and prev_close:
                change = float(last_price) - float(prev_close)
                change_percent = (change / float(prev_close)) * 100 if float(prev_close) else 0
        except Exception:
            pass

        # If prices are still zero/None, fallback to 5d chart
        try:
            if not last_price or float(last_price) == 0:
                data_5d = _retry(lambda: fetch_chart('5d', '1d'), attempts=2)
                result = (data_5d or {}).get('chart', {}).get('result', [])
                if result:
                    r0 = result[0]
                    # Prefer currency from Yahoo meta if present
                    try:
                        meta_cur2 = ((r0 or {}).get('meta') or {}).get('currency')
                        if meta_cur2:
                            currency_code = str(meta_cur2).upper()
                    except Exception:
                        pass
                    quotes = ((r0.get('indicators') or {}).get('quote') or [{}])[0]
                    closes = quotes.get('close') or []
                    opens = quotes.get('open') or []
                    highs = quotes.get('high') or []
                    lows = quotes.get('low') or []
                    vals = [c for c in closes if c is not None]
                    if vals:
                        last_price = float(vals[-1])
                        if (not prev_close or float(prev_close) == 0) and len(vals) >= 2:
                            prev_close = float(vals[-2])
                    if opens:
                        oo = [o for o in opens if o is not None]
                        if oo:
                            open_price = float(oo[-1])
                    if highs:
                        hh = [h for h in highs if h is not None]
                        if hh:
                            day_high = float(hh[-1])
                    if lows:
                        ll = [l for l in lows if l is not None]
                        if ll:
                            day_low = float(ll[-1])
        except Exception:
            pass

        # Recompute change metrics using final last_price and prev_close
        try:
            if last_price and prev_close:
                change = float(last_price) - float(prev_close)
                change_percent = (change / float(prev_close)) * 100 if float(prev_close) else 0
        except Exception:
            pass

        # If we still have no valid pricing info, return an explicit error
        if not last_price or float(last_price) == 0:
            logger.error(f"No valid price data returned for {symbol}")
            return jsonify({'error': 'No price data available for this symbol at the moment.'}), 502

        response = {
            'symbol': symbol,
            'name': quote.get('shortName') or quote.get('longName') or symbol,
            'price': float(last_price or 0),
            'change': float(change or 0),
            'changePercent': float(change_percent or 0),
            'marketCap': float(market_cap or 0),
            'volume': int(volume or 0),
            'peRatio': float(quote.get('trailingPE') or 0) if isinstance(quote.get('trailingPE'), (int, float)) else 0,
            'dividendYield': float(quote.get('trailingAnnualDividendYield') or 0) if isinstance(quote.get('trailingAnnualDividendYield'), (int, float)) else 0,
            'fiftyTwoWeekHigh': float(quote.get('fiftyTwoWeekHigh') or 0),
            'fiftyTwoWeekLow': float(quote.get('fiftyTwoWeekLow') or 0),
            'open': float(open_price or 0),
            'high': float(day_high or 0),
            'low': float(day_low or 0),
            'previousClose': float(prev_close or 0),
            'exchange': derived_exchange or quote.get('fullExchangeName') or quote.get('exchange') or 'N/A',
            'industry': 'N/A',
            'logo': '',
            'history': history,
            'currency': currency_code,
        }
        # Write to short-lived cache
        _cache_set(symbol, response)
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error fetching stock {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict')
def predict_price():
    """Predict next-day price using selected model over a date range.

    Query params:
    - symbol: required
    - from: optional ISO date YYYY-MM-DD
    - to: optional ISO date YYYY-MM-DD
    - model: one of ['lr','rf','arima']
    """
    try:
        symbol = request.args.get('symbol', '').strip()
        if not symbol:
            return jsonify({'error': 'symbol is required'}), 400

        model = (request.args.get('model', 'lr') or 'lr').lower()
        date_from = request.args.get('from')
        date_to = request.args.get('to')

        # Determine currency by symbol suffix
        currency_code = 'USD'
        try:
            suffix_map = {
                '.NS': 'INR', '.BO': 'INR', '.L': 'GBP', '.TO': 'CAD', '.HK': 'HKD',
                '.AX': 'AUD', '.SZ': 'CNY', '.SS': 'CNY', '.T': 'JPY', '.PA': 'EUR', '.F': 'EUR', '.DE': 'EUR'
            }
            for suf, cur in suffix_map.items():
                if symbol.upper().endswith(suf):
                    currency_code = cur
                    break
        except Exception:
            pass

        end = datetime.strptime(date_to, '%Y-%m-%d') if date_to else datetime.now()
        start = datetime.strptime(date_from, '%Y-%m-%d') if date_from else (end - timedelta(days=365))

        # Normalize training window: ensure we have past data ending no later than today
        train_end = min(datetime.now(), end)
        # If provided start is after train_end (e.g., future), fall back to 365d window
        train_start = start if start < train_end else (train_end - timedelta(days=365))

        # Fetch history via Yahoo chart API (no yfinance)
        def fetch_chart_range(sym: str, start_dt: datetime, end_dt: datetime):
            import time as _t
            url = f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}'
            params = {
                'period1': int(_t.mktime(start_dt.timetuple())),
                'period2': int(_t.mktime(end_dt.timetuple())),
                'interval': '1d'
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Referer': 'https://finance.yahoo.com/'
            }
            s = get_http_session() or requests
            r = s.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            return r.json()

        data = fetch_chart_range(symbol, train_start, train_end)
        result = (data.get('chart') or {}).get('result') or []
        if not result:
            return jsonify({'error': 'No historical data found'}), 404
        res = result[0]
        # Prefer currency from Yahoo meta if present
        try:
            meta_cur = ((res or {}).get('meta') or {}).get('currency')
            if meta_cur:
                currency_code = str(meta_cur).upper()
        except Exception:
            pass
        ts = res.get('timestamp') or []
        quotes = ((res.get('indicators') or {}).get('quote') or [{}])[0]
        closes = quotes.get('close') or []
        import pandas as _pd
        rows = []
        for i, t in enumerate(ts):
            if i >= len(closes) or closes[i] is None:
                continue
            try:
                dt = datetime.utcfromtimestamp(int(t))
            except Exception:
                continue
            rows.append({'Date': dt, 'Close': float(closes[i])})
        if not rows:
            return jsonify({'error': 'No historical data available to train the model'}), 404
        hist_df = _pd.DataFrame(rows).set_index('Date')
        df = hist_df[['Close']].dropna().rename(columns={'Close': 'close'})
        # If very little data, broaden window once more best-effort
        if len(df) < 30:
            try:
                alt_start = train_end - timedelta(days=540)
                data2 = fetch_chart_range(symbol, alt_start, train_end)
                result2 = (data2.get('chart') or {}).get('result') or []
                if result2:
                    res2 = result2[0]
                    # Check currency again
                    try:
                        meta_cur2 = ((res2 or {}).get('meta') or {}).get('currency')
                        if meta_cur2:
                            currency_code = str(meta_cur2).upper()
                    except Exception:
                        pass
                    ts2 = res2.get('timestamp') or []
                    quotes2 = ((res2.get('indicators') or {}).get('quote') or [{}])[0]
                    closes2 = quotes2.get('close') or []
                    rows2 = []
                    for i, t in enumerate(ts2):
                        if i >= len(closes2) or closes2[i] is None:
                            continue
                        try:
                            dt2 = datetime.utcfromtimestamp(int(t))
                        except Exception:
                            continue
                        rows2.append({'Date': dt2, 'Close': float(closes2[i])})
                    if rows2:
                        hist_df2 = _pd.DataFrame(rows2).set_index('Date')
                        df = hist_df2[['Close']].dropna().rename(columns={'Close': 'close'})
            except Exception:
                pass
        if len(df) < 10:
            return jsonify({'error': 'Not enough historical data to train. Try a different symbol or earlier From date.'}), 400

        # Prepare supervised learning dataset: features are lagged returns
        data = df.copy()
        data['return'] = data['close'].pct_change()
        for lag in range(1, 6):
            data[f'return_lag_{lag}'] = data['return'].shift(lag)
        data = data.dropna().copy()

        X = data[[f'return_lag_{i}' for i in range(1, 6)]]
        y = data['return']

        # Determine forecast horizon (business days) if 'to' is in the future
        last_hist_date = df.index[-1]
        target_date = end
        steps = 1
        try:
            if target_date.date() > last_hist_date.date():
                import pandas as _pd
                bdays = _pd.bdate_range(start=last_hist_date + pd.Timedelta(days=1), end=target_date)
                steps = max(1, len(bdays))
        except Exception:
            steps = 1

        # Train/test split: last 10% for evaluation
        split_idx = int(len(X) * 0.9)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        rmse = None
        predicted_price = None
        used_model = model

        last_close = float(df['close'].iloc[-1])

        # Prepare chart data
        chart_data = {
            'labels': [],
            'actual_prices': [],
            'predicted_prices': [],
            'dates': []
        }

        if model == 'lr':
            from sklearn.linear_model import LinearRegression
            m = LinearRegression()
            m.fit(X_train, y_train)
            import numpy as np
            y_pred = m.predict(X_test)
            rmse = float(((y_pred - y_test.values) ** 2).mean() ** 0.5)
            # Iterative multi-step forecast using returns
            lag_returns = [float(data[f'return_lag_{i}'].iloc[-1]) for i in range(1, 6)]
            lag_returns = lag_returns[::-1]  # order: most recent first
            price = last_close
            for _ in range(steps):
                features = np.array(lag_returns[:5][::-1], dtype=float).reshape(1, -1)
                next_ret = float(m.predict(features)[0])
                price *= (1.0 + next_ret)
                lag_returns.insert(0, next_ret)
            predicted_price = price
        elif model == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(n_estimators=200, random_state=42)
            m.fit(X_train, y_train)
            import numpy as np
            y_pred = m.predict(X_test)
            rmse = float(((y_pred - y_test.values) ** 2).mean() ** 0.5)
            lag_returns = [float(data[f'return_lag_{i}'].iloc[-1]) for i in range(1, 6)]
            lag_returns = lag_returns[::-1]
            price = last_close
            for _ in range(steps):
                features = np.array(lag_returns[:5][::-1], dtype=float).reshape(1, -1)
                next_ret = float(m.predict(features)[0])
                price *= (1.0 + next_ret)
                lag_returns.insert(0, next_ret)
            predicted_price = price
        else:
            # ARIMA on close prices
            used_model = 'arima'
            try:
                from statsmodels.tsa.arima.model import ARIMA
                series = df['close']
                # Simple order; could be tuned
                model_arima = ARIMA(series, order=(5,1,0))
                model_fit = model_arima.fit()
                forecast = model_arima.fit().forecast(steps=steps)
                predicted_price = float(forecast.iloc[-1])
                rmse = 0.0
            except Exception as e:
                logger.error(f'ARIMA failed: {e}')
                return jsonify({'error': 'ARIMA failed'}), 500

        # Generate chart data for visualization
        try:
            # Historical data for chart
            for date, close_price in df['close'].items():
                chart_data['labels'].append(date.strftime('%Y-%m-%d'))
                chart_data['actual_prices'].append(float(close_price))
                chart_data['dates'].append(date.strftime('%Y-%m-%d'))
            
            # Add prediction points
            if steps > 1:
                # For multi-step predictions, generate intermediate dates
                import pandas as _pd
                future_dates = _pd.bdate_range(start=last_hist_date + pd.Timedelta(days=1), end=target_date)
                for i, future_date in enumerate(future_dates):
                    chart_data['labels'].append(future_date.strftime('%Y-%m-%d'))
                    chart_data['dates'].append(future_date.strftime('%Y-%m-%d'))
                    # For now, just show the final prediction
                    if i == len(future_dates) - 1:
                        chart_data['predicted_prices'].append(predicted_price)
                    else:
                        chart_data['predicted_prices'].append(None)
            else:
                # Single step prediction
                next_date = last_hist_date + pd.Timedelta(days=1)
                chart_data['labels'].append(next_date.strftime('%Y-%m-%d'))
                chart_data['actual_prices'].append(None)
                chart_data['predicted_prices'].append(predicted_price)
                chart_data['dates'].append(next_date.strftime('%Y-%m-%d'))
        except Exception as e:
            logger.error(f'Error generating chart data: {e}')
            chart_data = None

        return jsonify({
            'model': used_model,
            'predicted_price': round(float(predicted_price), 4) if predicted_price is not None else None,
            'rmse': round(float(rmse), 6) if rmse is not None else None,
            'steps_ahead': int(steps),
            'target_date': target_date.strftime('%Y-%m-%d'),
            'note': 'Educational prediction. Do not use for trading.',
            'chart_data': chart_data,
            'currency': currency_code,
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/chart-data/<symbol>')
def get_chart_data(symbol):
    """Get historical chart data for a stock symbol."""
    try:
        # Fetch historical data for the last 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        def fetch_chart_range(sym: str, start_dt: datetime, end_dt: datetime):
            import time as _t
            url = f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}'
            params = {
                'period1': int(_t.mktime(start_dt.timetuple())),
                'period2': int(_t.mktime(end_dt.timetuple())),
                'interval': '1d'
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/plain, */*',
                'Referer': 'https://finance.yahoo.com/'
            }
            s = get_http_session() or requests
            r = s.get(url, params=params, headers=headers, timeout=15)
            r.raise_for_status()
            return r.json()

        data = fetch_chart_range(symbol, start_date, end_date)
        result = (data.get('chart') or {}).get('result') or []
        if not result:
            return jsonify({'error': 'No historical data found'}), 404
            
        res = result[0]
        ts = res.get('timestamp') or []
        quotes = ((res.get('indicators') or {}).get('quote') or [{}])[0]
        closes = quotes.get('close') or []
        volumes = quotes.get('volume') or []
        
        chart_data = {
            'labels': [],
            'prices': [],
            'volumes': []
        }
        
        for i, t in enumerate(ts):
            if i >= len(closes) or closes[i] is None:
                continue
            try:
                dt = datetime.utcfromtimestamp(int(t))
                chart_data['labels'].append(dt.strftime('%Y-%m-%d'))
                chart_data['prices'].append(float(closes[i]))
                chart_data['volumes'].append(int(volumes[i]) if i < len(volumes) and volumes[i] is not None else 0)
            except Exception:
                continue
                
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Error fetching chart data for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'message': 'The requested resource was not found on this server.'
    }), 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An internal server error occurred.'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)
