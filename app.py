from flask import Flask, request, jsonify
from chatbot import chatbot_response  # Import the chatbot response function

app = Flask(__name__)

# Route to handle the root URL for testing
@app.route('/')
def index():
    return "Welcome to the Chatbot API!"

# Endpoint for chatbot interaction (POST /chatbot)
@app.route('/chatbot', methods=['POST'])
def handle_chatbot():
    data = request.get_json()
    query = data.get('message')  # Extract user message from the request
    if query:
        response = chatbot_response(query)  # Get the chatbot response
        return jsonify({"reply": response})
    else:
        return jsonify({"reply": "No query received."}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
