### Government Scheme Chatbot

This project is a chatbot designed to provide information about various government schemes. It uses natural language processing (NLP) and a FAISS-based search engine to retrieve relevant information from a
dataset of government schemes. The chatbot is built using Flask for the backend, and it leverages the LLaMA model via the Groq API for generating responses.

### Features

**Natural Language Understanding**: The chatbot can understand user queries related to government schemes and provide relevant information.

**FAISS-based Search: Uses FAISS**: (Facebook AI Similarity Search) to quickly find the most relevant schemes based on user queries.

**LLaMA Integration**: Leverages the LLaMA model for generating natural language responses.

**User-friendly Interface**: A simple and intuitive web interface for interacting with the chatbot.

### Technologies Used

**Backend**: Flask (Python)

**Frontend**: HTML, CSS, JavaScript

**NLP Model**: BERT (via Hugging Face Transformers)

**Search Engine**: FAISS

**LLM**: LLaMA (via Groq API)

### Installation

1. **Clone the repository**
   ```
   git clone https://github.com/your-username/government-scheme-chatbot.git
   cd government-scheme-chatbot

2. **Set up a virtual environment (optional but recommended)**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. **Install the required Python packages**
   ```
   pip install -r requirements.txt

4. **Set up environment variables**

   Create a .env file in the root directory and add your Groq API key:

   GROQ_API_KEY=your_groq_api_key_here

5. **Run the Chatbot Application**
   ```
   python chatbot.py

6. **Access the chatbot**

   Open your web browser and navigate to http://localhost:5000 to interact with the chatbot.

### Usage

**Ask Questions**: Type your questions related to government schemes in the input box and press "Send".

**Get Responses**: The chatbot will process your query and provide relevant information about the schemes.

### Project Structure
```
government-scheme-chatbot/
├── app.py                # Flask application entry point
├── chatbot.py            # Chatbot logic and NLP processing
├── static/               # Static files (CSS, JS, JSON)
│   ├── css/
│   │   └── style.css     # Stylesheet for the chatbot interface
│   ├── js/
│   │   └── chatbot.js    # JavaScript for handling chatbot interactions
│   └── Cleaned_Schemes.json  # JSON file containing government schemes data
├── templates/            # HTML templates
│   └── index.html        # Main chatbot interface
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .env                  # Environment variables (not included in the repo)
```
### Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps: 

1. **Fork the repository**

2. **Create a new branch**
   ```
   git checkout -b feature/YourFeatureName

3. **Commit your changes**
   ```
   git commit -m 'Add some feature

5. **Push to the branch**:
   ```
   git push origin feature/YourFeatureName

6. **Open a pull request**

### Acknowledgments

Hugging Face for the BERT model and tokenizer.

Facebook AI for the FAISS library.

Groq for providing access to the LLaMA model.

### Contact

If you have any questions or suggestions, feel free to reach out:

**My email**: mohamedashik027@gmail.com

**Project Link**: https://github.com/your-username/government-scheme-chatbot
