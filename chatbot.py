import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, request, jsonify, render_template
import json
import numpy as np
import faiss
import torch
from transformers import BertTokenizer, BertModel
from groq import Groq
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load dataset (Cleaned_Schemes.json)
with open("static/Cleaned_Schemes.json", "r") as f:
    schemes_data = json.load(f)

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
bert_model = BertModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Encode text using BERT
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Create FAISS index
def create_faiss_index():
    combined_texts = [f"{scheme['Scheme Name']}. {scheme['Description']}" for scheme in schemes_data]
    embeddings = np.vstack([get_bert_embedding(text) for text in combined_texts]).astype("float32")
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    return faiss_index, schemes_data

# Search FAISS index for similar schemes
def search_faiss(query, faiss_index, schemes_data, top_k=5):
    query_embedding = get_bert_embedding(query).astype("float32").reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = [schemes_data[idx] for idx in indices[0]]
    return results, distances[0]

# Combine and rank FAISS results
def combine_results(faiss_results, query_lower):
    faiss_weight = 3
    exact_match_weight = 5
    partial_match_weight = 2

    scores = {}

    for result in faiss_results:
        scheme_name = result["Scheme Name"].lower()
        scores[scheme_name] = scores.get(scheme_name, 0) + faiss_weight

    sorted_schemes = sorted(scores.items(), key=lambda x: -x[1])
    combined_results = [
        res
        for res in faiss_results
        if res["Scheme Name"].lower() in dict(sorted_schemes)
    ]

    return combined_results

# Initialize FAISS index and schemes data at startup
faiss_index, schemes_data = create_faiss_index()

# Detect intent of the query
def detect_intent(query):
    scheme_keywords = ["scheme", "scholarship", "internship", "benefits", "eligibility", "application", "documents", "government"]
    if any(keyword in query.lower() for keyword in scheme_keywords):
        return "scheme"
    else:
        return "general"

# Check if query is specifically about a scholarship
def is_scholarship_query(query):
    query_lower = query.lower()
    return "scholarship" in query_lower

# Check if a query is likely asking for a specific scheme
def is_specific_scheme_query(query):
    specific_indicators = ["tell me about", "information on", "details of", "what is", "how to apply for", "eligibility for", "benefits of", "can you tell me about"]
    return any(indicator in query.lower() for indicator in specific_indicators)

def generate_llama_response(scheme, query):
    """
    Generates a natural language response using the LLaMA model.
    """
    scheme_name = scheme.get('Scheme Name', 'Unknown Scheme')
    description = scheme.get('Description', 'Description not available.')
    eligibility = scheme.get('Eligibility', 'Eligibility criteria not specified.')
    benefits = scheme.get('Benefits', 'Benefits not mentioned.')
    required_documents = scheme.get('Required Documents', 'Required documents not specified.')

    # Construct a prompt for the LLaMA model
    prompt = (
        f"You are a helpful assistant that provides information about government schemes. "
        f"Here is some information about the '{scheme_name}' scheme:\n\n"
        f"**Description**: {description}\n"
        f"**Eligibility**: {eligibility}\n"
        f"**Benefits**: {benefits}\n"
        f"**Required Documents**: {required_documents}\n\n"
        f"Based on this information, answer the following user query in a clear and concise manner:\n"
        f"**User Query**: {query}\n\n"
        f"**Response**:"
    )

    # Use the LLaMA model to generate a response
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    response = chat_completion.choices[0].message.content

    return response

@app.route('/chatbot', methods=['POST'])
def handle_chatbot():
    data = request.get_json()
    query = data.get('message')
    if query:
        try:
            intent = detect_intent(query)
            logging.debug(f"Detected intent: {intent}")

            if intent == "scheme":
                query_lower = query.lower()
                logging.debug(f"Processing scheme-related query: {query}")
                
                # Special handling for queries containing scheme names that don't exist
                # Extract potential scheme name from query
                words = query_lower.split()
                potential_scheme_name = None
                
                # Check for patterns like "xyz scheme" or "xyz scholarship"
                for i, word in enumerate(words):
                    if i > 0 and (word == "scheme" or word == "scholarship"):
                        potential_scheme_name = words[i-1]
                        break
                
                # If we identified a potential scheme name, check if it exists in our data
                scheme_exists = False
                if potential_scheme_name:
                    for scheme in schemes_data:
                        if potential_scheme_name.lower() in scheme["Scheme Name"].lower():
                            scheme_exists = True
                            break
                
                # If we have a scheme name and it doesn't exist in our data, return not found message
                if potential_scheme_name and not scheme_exists:
                    logging.debug(f"Potential scheme '{potential_scheme_name}' not found in database")
                    
                    prompt = (
                        "You are a helpful assistant providing information about government schemes. "
                        "Your database does NOT contain information about this specific scheme or scholarship. "
                        "Respond with a brief message stating only that you don't have information about this "
                        "specific scheme in your database. DO NOT suggest or mention ANY other schemes or scholarships."
                    )
                    
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                    )
                    response = chat_completion.choices[0].message.content
                    return jsonify({"reply": response})
                
                # Continue with normal processing for other queries
                # Search for relevant schemes using FAISS
                faiss_results, distances = search_faiss(query, faiss_index, schemes_data)
                logging.debug(f"FAISS results: {faiss_results}")
                logging.debug(f"FAISS distances: {distances}")

                # Set a threshold for what counts as a valid match
                distance_threshold = 25.0  # Adjust this value based on testing
                
                # Check if the best match is too far (irrelevant)
                if distances[0] > distance_threshold:
                    prompt = (
                        "You are a helpful assistant providing information about government schemes. "
                        f"The user query is: '{query}'. "
                        "Your response must be ONLY: 'The requested scheme is not available in database.' "
                        "Do not add any additional information or suggestions."
                    )
                    
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="llama-3.3-70b-versatile",
                    )
                    response = "The requested scheme is not available in database."
                else:
                    # Combine and rank the results
                    combined_results = combine_results(faiss_results, query_lower)
                    logging.debug(f"Combined results: {combined_results}")

                    if combined_results:
                        # Match is close enough, generate a response
                        best_match = combined_results[0]
                        response = generate_llama_response(best_match, query)
                        logging.debug(f"Generated response: {response}")
                    else:
                        response = "The requested scheme is not available in database."
            else:
                # Handle general conversation
                logging.debug(f"Processing general conversation query: {query}")

                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": query}],
                    model="llama-3.3-70b-versatile",
                )
                response = chat_completion.choices[0].message.content
                logging.debug(f"Groq API response: {response}")

            return jsonify({"reply": response})
        except Exception as e:
            logging.error(f"Error processing query: {e}", exc_info=True)
            return jsonify({"reply": "Sorry, something went wrong. Please try again."}), 500
    else:
        return jsonify({"reply": "No query received."}), 400
# Serve the chatbot interface
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)