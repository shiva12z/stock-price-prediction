from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World! This is a test Flask server."

@app.route('/api/test')
def test_api():
    return {"status": "success", "message": "API is working!"}

if __name__ == '__main__':
    print("Starting test Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
