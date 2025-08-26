import requests
import sys

def test_search():
    try:
        # Test search endpoint
        print("Testing search endpoint...")
        response = requests.get(
            'http://localhost:5000/api/search',
            params={'q': 'AAPL'},
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
        
        # Test quote endpoint
        print("\nTesting quote endpoint...")
        response = requests.get(
            'http://localhost:5000/api/quote/AAPL',
            timeout=10
        )
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure the Flask server is running (run 'python app.py' in a terminal)")
        print("2. Check if port 5000 is available (netstat -ano | findstr :5000)")
        print("3. Verify your internet connection")
        print("4. Check if the Finnhub API key is set in the .env file")
        print("5. Look for any error messages in the Flask server console")

if __name__ == "__main__":
    test_search()
